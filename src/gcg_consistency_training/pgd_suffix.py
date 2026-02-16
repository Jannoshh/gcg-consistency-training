"""PGD suffix optimization for anti-sycophancy (Geisler et al. 2024).

Optimizes a discrete adversarial suffix via projected gradient descent on a
continuous relaxation. Each suffix position maintains a distribution over the
vocabulary on the probability simplex. Tsallis q=2 entropy projection is
annealed to gradually force discrete (one-hot) solutions.

Reference: https://arxiv.org/abs/2402.09154
Implementation adapted from: https://github.com/sigeisler/reinforce-attacks-llms
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from gcg_consistency_training.config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bct_data(path: str | Path) -> list[dict]:
    """Load BCT sycophancy JSONL data."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Projection algorithms (extracted from reinforce-attacks-llms)
# ---------------------------------------------------------------------------

def simplex_sort_projection(values: torch.Tensor) -> torch.Tensor:
    """Project onto probability simplex via sorting.

    From "Large-scale Multiclass SVM Training via Euclidean Projection
    onto the Simplex" (Blondel et al., ICPR 2014).

    Args:
        values: shape (batch, dims) — values to project.

    Returns:
        Projected values on the simplex.
    """
    b, d = values.shape
    cat_indices = torch.arange(d, device=values.device)
    batch_indices = torch.arange(b, device=values.device)

    values = torch.clamp_min(values, 0.0)
    values_sorted = -(-values).sort(-1).values
    values_cumulative = torch.cumsum(values_sorted, dim=-1) - 1
    condition = values_sorted - values_cumulative / (cat_indices + 1) > 0
    rho = torch.count_nonzero(condition, dim=-1)
    theta = values_cumulative[batch_indices, rho - 1] / rho
    values = torch.clamp_min(values - theta[:, None], 0.0)
    return values


def project_simplex(values: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Full simplex projection with degenerate-case handling.

    Projects each row of `values` (shape [..., vocab]) onto the probability
    simplex, handling the case where all values are zero.
    """
    orig_shape = values.shape
    values = values.reshape(-1, orig_shape[-1])

    exceeds_budget = torch.clamp(values, 0, 1).sum(-1) > 1
    if exceeds_budget.any():
        values[exceeds_budget] = simplex_sort_projection(values[exceeds_budget])
        values[~exceeds_budget] = torch.clamp(values[~exceeds_budget], min=0, max=1)
    else:
        values = torch.clamp(values, min=0, max=1)

    # Handle degenerate case: all zeros
    all_zero = torch.isclose(values.sum(-1, keepdim=True), torch.tensor(0.0))
    values += all_zero * torch.rand_like(values)
    values = values / torch.clamp_min(values.sum(-1, keepdim=True), eps)

    return values.reshape(orig_shape)


def tsallis_q2_projection(
    values: torch.Tensor,
    entropy_factor: float | torch.Tensor,
    disallowed_tokens: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Tsallis q=2 entropy projection for sparsity.

    Projects values onto a sphere within the simplex that constrains
    the Tsallis entropy. Higher entropy_factor → more one-hot (sparse).

    Args:
        values: shape (..., vocab) on the simplex.
        entropy_factor: in (0, 1], controls sparsity.
        disallowed_tokens: token indices to keep at zero.
        eps: numerical stability.
    """
    orig_shape = values.shape
    values = values.reshape(-1, orig_shape[-1])

    normal = torch.ones(values.shape[-1], device=values.device)
    if disallowed_tokens is not None:
        normal[disallowed_tokens] = 0

    # Exclude already-zero positions from the projection
    is_close_to_zero = torch.isclose(values, torch.tensor(0.0))
    normal_expanded = normal[None].expand_as(is_close_to_zero).clone()
    normal_expanded[is_close_to_zero] = 0
    normal_expanded = normal_expanded / torch.clamp_min(
        normal_expanded.norm(dim=-1, keepdim=True), eps
    )

    non_zero = normal_expanded > 0
    d = non_zero.sum(-1)
    target_entropy = (1 - entropy_factor) * (d - 1) / d
    center = (1.0 / d[..., None]) * non_zero

    dist_to_hyperplane = (values * normal_expanded).sum(-1)
    projection_radius = torch.sqrt(
        torch.clamp(1 - target_entropy - dist_to_hyperplane**2, 0)
    )[..., None]

    direction = values - center
    direction_norm = torch.clamp_min(
        torch.linalg.norm(direction, dim=-1, keepdim=True), eps
    )
    exceeds_budget = (direction_norm < projection_radius)[..., 0]

    if exceeds_budget.any():
        values_ = projection_radius / direction_norm * direction + center
        # Re-project onto simplex to ensure feasibility
        values_[exceeds_budget] = project_simplex(values_[exceeds_budget])
        values = torch.where(exceeds_budget[..., None], values_, values)

    return values.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Non-ASCII token list (from nanogcg)
# ---------------------------------------------------------------------------

def get_nonascii_toks(tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """Get indices of tokens that contain non-ASCII characters."""

    def is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    nonascii = []
    for i in range(tokenizer.vocab_size):
        tok = tokenizer.decode([i])
        if not is_ascii(tok):
            nonascii.append(i)
    return torch.tensor(nonascii, dtype=torch.long)


# ---------------------------------------------------------------------------
# PGD Suffix Optimizer
# ---------------------------------------------------------------------------

class PGDSuffixOptimizer:
    """PGD optimization for discrete suffix tokens.

    Maintains embedding_factors: [suffix_len, vocab_size] on probability simplex.
    Each step: soft forward → CE loss → backward → Adam step → simplex project → entropy project.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Config,
    ):
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.config = config
        self.pgd = config.pgd
        self.device = model.device

        self.embed_matrix = model.get_input_embeddings().weight  # [vocab, d_model]
        self.vocab_size = self.embed_matrix.shape[0]
        self.d_model = self.embed_matrix.shape[1]

        # Disallowed tokens mask
        self.disallowed_tokens = None
        if not self.pgd.allow_non_ascii:
            self.disallowed_tokens = get_nonascii_toks(tokenizer).to(self.device)

        self.eps = torch.finfo(torch.float32).eps

    def init_embedding_factors(self) -> torch.Tensor:
        """Initialize embedding factors randomly on the simplex."""
        shape = (self.pgd.suffix_length, self.vocab_size)
        factors = torch.rand(shape, dtype=torch.float32, device=self.device)

        if self.disallowed_tokens is not None:
            factors[:, self.disallowed_tokens] = 0.0

        # Normalize to simplex
        factors = factors / torch.clamp_min(factors.sum(-1, keepdim=True), self.eps)
        return factors

    def soft_forward(
        self,
        prompt_ids: torch.Tensor,
        factors: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with soft suffix embeddings.

        Computes: prompt_embeds ++ (factors @ embed_matrix) ++ target_embeds → model → CE loss
        on target token positions.

        Args:
            prompt_ids: [batch, prompt_len] token IDs for the prompt.
            factors: [suffix_len, vocab_size] continuous relaxation.
            target_ids: [batch, target_len] token IDs for the target answer.

        Returns:
            CE loss averaged over batch.
        """
        batch_size = prompt_ids.shape[0]

        # Upcast embed_matrix to float32 for matmul so gradients flow through factors
        embed_f32 = self.embed_matrix.float()

        # Embed prompt tokens (in model dtype, then upcast)
        prompt_embeds = embed_f32[prompt_ids]  # [batch, prompt_len, d_model]

        # Soft suffix embeddings: factors @ embed_matrix (both fp32)
        suffix_embeds = factors @ embed_f32  # [suffix_len, d_model]
        suffix_embeds = suffix_embeds.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, suffix_len, d_model]

        # Embed target tokens
        target_embeds = embed_f32[target_ids]  # [batch, target_len, d_model]

        # Concatenate: prompt ++ suffix ++ target
        inputs_embeds = torch.cat(
            [prompt_embeds, suffix_embeds, target_embeds], dim=1
        )

        # Cast back to model dtype for the forward pass
        inputs_embeds = inputs_embeds.to(self.embed_matrix.dtype)

        # Forward pass
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = outputs.logits  # [batch, seq_len, vocab]

        # CE loss on target positions only
        # Target starts at position (prompt_len + suffix_len)
        prompt_len = prompt_ids.shape[1]
        suffix_len = self.pgd.suffix_length
        target_start = prompt_len + suffix_len
        target_len = target_ids.shape[1]

        # Shift: logits at position t predict token at position t+1
        # So logits at [target_start-1 : target_start+target_len-1] predict target tokens
        pred_logits = logits[:, target_start - 1 : target_start + target_len - 1, :]
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="mean",
        )

        return loss

    def discrete_forward(
        self,
        prompt_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with discrete suffix tokens (for monitoring)."""
        batch_size = prompt_ids.shape[0]
        suffix_expanded = suffix_ids.unsqueeze(0).expand(batch_size, -1)
        input_ids = torch.cat([prompt_ids, suffix_expanded, target_ids], dim=1)

        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits

        prompt_len = prompt_ids.shape[1]
        suffix_len = suffix_ids.shape[0]
        target_start = prompt_len + suffix_len
        target_len = target_ids.shape[1]

        pred_logits = logits[:, target_start - 1 : target_start + target_len - 1, :]
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="mean",
        )
        return loss

    def kl_forward(
        self,
        clean_prompt_ids: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(P_clean_no_suffix || P_clean_with_suffix).

        Measures how much the suffix changes the model's next-token distribution
        on clean (unbiased) prompts. We want this to be small so the suffix
        doesn't degrade performance on normal inputs.

        Args:
            clean_prompt_ids: [batch, prompt_len] token IDs for clean prompts.
            factors: [suffix_len, vocab_size] continuous relaxation.

        Returns:
            KL divergence averaged over batch and positions.
        """
        batch_size = clean_prompt_ids.shape[0]
        embed_f32 = self.embed_matrix.float()
        model_dtype = self.embed_matrix.dtype

        # --- Reference: clean prompt without suffix ---
        with torch.no_grad():
            ref_outputs = self.model(
                input_ids=clean_prompt_ids, use_cache=False
            )
            # Logits at the last chat-template token (predicts first response token)
            ref_logits = ref_outputs.logits[:, -1, :].float()  # [batch, vocab]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # --- With suffix: clean prompt ++ soft suffix ---
        prompt_embeds = embed_f32[clean_prompt_ids]
        suffix_embeds = (factors @ embed_f32).unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(model_dtype)

        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        # Last position (end of suffix) predicts first response token
        suffixed_logits = outputs.logits[:, -1, :].float()  # [batch, vocab]
        suffixed_log_probs = F.log_softmax(suffixed_logits, dim=-1)

        # KL(P_ref || P_suffixed) = sum(P_ref * (log P_ref - log P_suffixed))
        kl = F.kl_div(suffixed_log_probs, ref_log_probs, log_target=True, reduction="batchmean")

        return kl

    def prepare_clean_batch(
        self,
        batch: list[dict],
    ) -> torch.Tensor:
        """Prepare clean (unbiased) prompts from BCT data for KL computation."""
        prompt_texts = []
        for item in batch:
            prompt = item["clean_prompt"]
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(chat_text)

        encodings = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)

        return encodings.input_ids

    def discretize(self, factors: torch.Tensor) -> tuple[torch.Tensor, str]:
        """Convert continuous factors to discrete tokens via argmax.

        Returns (token_ids, decoded_string).
        """
        ids = factors.argmax(-1)  # [suffix_len]
        string = self.tokenizer.decode(ids, skip_special_tokens=True)

        # Re-encode to handle tokenizer artifacts
        re_ids = self.tokenizer.encode(string, add_special_tokens=False)
        re_ids = torch.tensor(re_ids, device=self.device)

        return re_ids, string

    def generate_reference_responses(
        self, data: list[dict], batch_size: int = 16
    ) -> list[str]:
        """Generate model's own responses on clean (unbiased) prompts.

        These serve as CE targets during training: we want the suffix to make
        the model produce the same response on biased prompts as it naturally
        gives on clean prompts (BCT approach).
        """
        logger.info(
            f"Generating reference responses for {len(data)} clean prompts "
            f"(batch_size={batch_size})..."
        )
        responses = [""] * len(data)
        answer_pattern = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)

        # Ensure left-padding for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for start in tqdm(range(0, len(data), batch_size), desc="Generating refs"):
            end = min(start + batch_size, len(data))
            batch = data[start:end]

            # Format all prompts in this batch
            formatted = []
            for item in batch:
                messages = [{"role": "user", "content": item["clean_prompt"]}]
                chat_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted.append(chat_text)

            # Tokenize with left-padding
            encodings = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                    max_new_tokens=self.config.model.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode each response
            prompt_len = encodings.input_ids.shape[1]
            for i in range(len(batch)):
                gen_ids = outputs[i, prompt_len:]
                response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Truncate at </answer>
                match = answer_pattern.search(response)
                if match:
                    response = response[:match.end()]

                responses[start + i] = response

        # Restore padding side
        self.tokenizer.padding_side = "right"

        logger.info(f"Generated {len(responses)} reference responses")
        return responses

    def prepare_batch(
        self,
        batch: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of BCT data for training.

        Returns (prompt_ids, target_ids) padded to same length.
        Uses pre-generated reference responses as targets if available.
        """
        prompt_texts = []
        target_texts = []

        for item in batch:
            # Use wrapped_prompt (contains sycophancy bias)
            prompt = item["wrapped_prompt"]

            # Format as chat message
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(chat_text)

            # Target: pre-generated reference response from clean prompt
            target = item.get("_reference_response", f"<answer>{item['correct_answer']}</answer>")
            target_texts.append(target)

        # Tokenize prompts (left-pad for batching)
        prompt_encodings = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)

        # Tokenize targets
        target_encodings = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)

        return prompt_encodings.input_ids, target_encodings.input_ids

    def get_entropy_factor(self, step: int) -> float:
        """Linearly anneal entropy factor from start to end."""
        if step >= self.pgd.entropy_anneal_steps:
            return self.pgd.entropy_factor_end
        t = step / max(self.pgd.entropy_anneal_steps, 1)
        return self.pgd.entropy_factor_start + t * (
            self.pgd.entropy_factor_end - self.pgd.entropy_factor_start
        )

    def evaluate_accuracy(
        self,
        data: list[dict],
        suffix_ids: torch.Tensor,
        max_samples: int = 200,
    ) -> dict:
        """Evaluate discrete suffix on BCT data.

        Tracks:
        - correct: model picked the right answer
        - sycophantic: model picked suggested_wrong (influenced by bias)
        - other: model picked something else or unparseable
        """
        correct = 0
        sycophantic = 0
        other = 0
        total = 0

        samples = data[:max_samples]
        answer_pattern = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)

        for item in samples:
            prompt = item["wrapped_prompt"]
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            prompt_ids = self.tokenizer.encode(
                chat_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            suffix_expanded = suffix_ids.unsqueeze(0)
            input_ids = torch.cat([prompt_ids, suffix_expanded], dim=1)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.model.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated_ids = output[0, input_ids.shape[1] :]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Parse answer
            match = answer_pattern.search(generated_text)
            if match:
                predicted = match.group(1).upper()
                correct_ans = item["correct_answer"].upper()
                suggested_wrong = item.get("suggested_wrong", "").upper()
                if predicted == correct_ans:
                    correct += 1
                elif predicted == suggested_wrong:
                    sycophantic += 1
                else:
                    other += 1
            else:
                other += 1
            total += 1

        sycophancy_rate = sycophantic / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        logger.info(
            f"Eval: accuracy={accuracy:.3f} sycophancy_rate={sycophancy_rate:.3f} "
            f"(correct={correct} sycophantic={sycophantic} other={other} total={total})"
        )

        return {
            "accuracy": accuracy,
            "sycophancy_rate": sycophancy_rate,
            "correct": correct,
            "sycophantic": sycophantic,
            "other": other,
            "total": total,
        }

    def train(
        self,
        train_data: list[dict],
        eval_data: list[dict] | None = None,
        wandb_run=None,
    ) -> dict:
        """Main PGD training loop.

        Returns dict with final suffix, losses, and metrics.
        """
        torch.manual_seed(self.pgd.seed)

        # Init embedding factors on simplex
        factors = self.init_embedding_factors()
        factors.requires_grad_(True)

        # Adam optimizer with cosine annealing
        optimizer = torch.optim.Adam([factors], lr=self.pgd.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.pgd.num_steps, eta_min=self.pgd.lr * 0.1
        )

        # Training state
        best_discrete_loss = float("inf")
        best_suffix_ids = None
        best_suffix_str = ""
        all_losses = []
        all_discrete_losses = []

        n_train = len(train_data)
        kl_weight = self.pgd.kl_weight
        all_kl_losses = []

        # Generate reference responses on clean prompts (BCT approach)
        # Check for cached references first
        ref_cache_path = Path("/results/ref_responses_cache.json")
        cache_key = f"n{n_train}_model{self.config.model.model_id}"
        cached = None
        if ref_cache_path.exists():
            import json as _json
            with open(ref_cache_path) as f:
                cached_data = _json.load(f)
            if cached_data.get("key") == cache_key and len(cached_data.get("responses", [])) == n_train:
                cached = cached_data["responses"]
                logger.info(f"Loaded {len(cached)} cached reference responses")

        if cached is not None:
            ref_responses = cached
        else:
            ref_responses = self.generate_reference_responses(train_data)
            # Save cache
            try:
                import json as _json
                ref_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ref_cache_path, "w") as f:
                    _json.dump({"key": cache_key, "responses": ref_responses}, f)
                logger.info(f"Cached {len(ref_responses)} reference responses to {ref_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache references: {e}")

        for i, resp in enumerate(ref_responses):
            train_data[i]["_reference_response"] = resp

        # Progressive discretization state
        frozen_mask = torch.zeros(
            self.pgd.suffix_length, dtype=torch.bool, device=self.device
        )  # True = position is frozen (discretized)
        n_frozen = 0
        prog_disc = self.pgd.progressive_discretize
        disc_warmup = self.pgd.discretize_warmup_steps
        disc_every = self.pgd.discretize_every
        if prog_disc and disc_every <= 0:
            # Auto: spread discretization evenly over remaining steps
            remaining_steps = self.pgd.num_steps - disc_warmup
            disc_every = max(1, remaining_steps // self.pgd.suffix_length)

        logger.info(
            f"Starting PGD optimization: {self.pgd.num_steps} steps, "
            f"suffix_len={self.pgd.suffix_length}, lr={self.pgd.lr}, "
            f"batch_size={self.pgd.batch_size}, train_samples={n_train}, "
            f"kl_weight={kl_weight}"
            f"{f', progressive_discretize every {disc_every} steps after {disc_warmup} warmup' if prog_disc else ''}"
        )

        for step in tqdm(range(self.pgd.num_steps), desc="PGD"):
            # 1. Sample batch
            indices = np.random.randint(0, n_train, size=self.pgd.batch_size)
            batch = [train_data[i] for i in indices]

            # 2. Get entropy factor for this step
            entropy_factor = self.get_entropy_factor(step)

            # 3. Prepare batch
            prompt_ids, target_ids = self.prepare_batch(batch)

            # 4. Soft forward + loss (two separate backward passes to save memory)
            optimizer.zero_grad()

            with torch.enable_grad():
                ce_loss = self.soft_forward(prompt_ids, factors, target_ids)
            ce_loss.backward()

            # KL regularization on clean prompts (separate backward to avoid OOM)
            kl_loss_val = torch.tensor(0.0, device=self.device)
            if kl_weight > 0:
                clean_prompt_ids = self.prepare_clean_batch(batch)
                with torch.enable_grad():
                    kl_loss_val = self.kl_forward(clean_prompt_ids, factors)
                    kl_scaled = kl_weight * kl_loss_val
                kl_scaled.backward()  # gradients accumulate on factors

            loss = ce_loss.detach() + kl_weight * kl_loss_val.detach()

            # 6. Gradient clipping (per-token norm, following reference)
            if factors.grad is not None:
                grad = factors.grad
                if self.disallowed_tokens is not None:
                    grad[:, self.disallowed_tokens] = 0.0
                # Zero gradients for frozen positions
                if n_frozen > 0:
                    grad[frozen_mask] = 0.0
                norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
                grad_clipped = torch.where(
                    norm > self.pgd.grad_clip,
                    self.pgd.grad_clip * grad / (norm + self.eps),
                    grad,
                )
                factors.grad.copy_(grad_clipped)

            # 7. Optimizer step
            optimizer.step()
            scheduler.step()

            # 8. Project onto simplex
            with torch.no_grad():
                factors.data.copy_(project_simplex(factors.data))

                # 9. Entropy projection (if annealed factor > 0)
                if entropy_factor > 0:
                    factors.data.copy_(
                        tsallis_q2_projection(
                            factors.data,
                            entropy_factor,
                            disallowed_tokens=self.disallowed_tokens,
                        )
                    )

                # Re-enforce frozen positions after projections
                if n_frozen > 0:
                    frozen_indices = torch.where(frozen_mask)[0]
                    for fi in frozen_indices:
                        tok = factors.data[fi].argmax().item()
                        factors.data[fi] = 0.0
                        factors.data[fi, tok] = 1.0

            # 10. Progressive discretization: freeze most confident position
            if (
                prog_disc
                and step >= disc_warmup
                and n_frozen < self.pgd.suffix_length
                and (step - disc_warmup) % disc_every == 0
            ):
                with torch.no_grad():
                    # Find most confident unfrozen position (lowest entropy)
                    unfrozen = ~frozen_mask
                    # Entropy per position: -sum(p * log(p))
                    p = factors.data[unfrozen]
                    log_p = torch.log(p + self.eps)
                    entropies = -(p * log_p).sum(-1)
                    # Pick the position with lowest entropy
                    unfrozen_indices = torch.where(unfrozen)[0]
                    best_local = entropies.argmin()
                    best_pos = unfrozen_indices[best_local].item()
                    # Snap to one-hot
                    best_token = factors.data[best_pos].argmax().item()
                    factors.data[best_pos] = 0.0
                    factors.data[best_pos, best_token] = 1.0
                    frozen_mask[best_pos] = True
                    n_frozen += 1
                    token_str = self.tokenizer.decode([best_token])
                    logger.info(
                        f"Step {step}: froze position {best_pos} -> "
                        f"token {best_token} '{token_str}' "
                        f"({n_frozen}/{self.pgd.suffix_length} frozen)"
                    )

            # 11. Discretize and compute discrete loss
            relaxed_loss = ce_loss.item()
            kl_loss_scalar = kl_loss_val.item()
            all_losses.append(relaxed_loss)
            all_kl_losses.append(kl_loss_scalar)

            with torch.no_grad():
                suffix_ids, suffix_str = self.discretize(factors)
                discrete_loss = self.discrete_forward(
                    prompt_ids, suffix_ids, target_ids
                ).item()

            all_discrete_losses.append(discrete_loss)
            relaxation_gap = (
                (discrete_loss - relaxed_loss) / max(abs(discrete_loss), self.eps)
            )

            # Track best
            if discrete_loss < best_discrete_loss:
                best_discrete_loss = discrete_loss
                best_suffix_ids = suffix_ids.clone()
                best_suffix_str = suffix_str

            # 11. Logging
            if wandb_run is not None:
                log_dict = {
                    "pgd/relaxed_loss": relaxed_loss,
                    "pgd/discrete_loss": discrete_loss,
                    "pgd/relaxation_gap": relaxation_gap,
                    "pgd/entropy_factor": entropy_factor,
                    "pgd/lr": scheduler.get_last_lr()[0],
                    "pgd/best_discrete_loss": best_discrete_loss,
                    "pgd/kl_loss": kl_loss_scalar,
                    "pgd/total_loss": loss.item(),
                    "pgd/step": step,
                    "pgd/n_frozen": n_frozen,
                }
                wandb_run.log(log_dict, step=step)

            if step % 50 == 0:
                kl_str = f" kl={kl_loss_scalar:.4f}" if kl_weight > 0 else ""
                logger.info(
                    f"Step {step}: relaxed={relaxed_loss:.4f} "
                    f"discrete={discrete_loss:.4f} gap={relaxation_gap:.3f} "
                    f"entropy_factor={entropy_factor:.3f} "
                    f"best_discrete={best_discrete_loss:.4f}"
                    f"{kl_str}"
                )
                logger.info(f"Current suffix: {suffix_str}")
                logger.info(f"Best suffix: {best_suffix_str}")

            # 12. Periodic evaluation
            if (
                eval_data is not None
                and self.pgd.eval_every > 0
                and (step + 1) % self.pgd.eval_every == 0
            ):
                eval_metrics = self.evaluate_accuracy(eval_data, suffix_ids)
                logger.info(
                    f"Step {step}: eval accuracy={eval_metrics['accuracy']:.3f} "
                    f"({eval_metrics['correct']}/{eval_metrics['total']})"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "pgd/eval_accuracy": eval_metrics["accuracy"],
                            "pgd/eval_correct": eval_metrics["correct"],
                        },
                        step=step,
                    )

        # Final evaluation with best suffix
        logger.info(f"\nTraining complete.")
        logger.info(f"Best discrete loss: {best_discrete_loss:.4f}")
        logger.info(f"Best suffix: {best_suffix_str}")

        final_eval = None
        if eval_data is not None and best_suffix_ids is not None:
            final_eval = self.evaluate_accuracy(
                eval_data, best_suffix_ids, max_samples=50
            )
            logger.info(
                f"Final eval: accuracy={final_eval['accuracy']:.3f} "
                f"sycophancy_rate={final_eval['sycophancy_rate']:.3f} "
                f"(correct={final_eval['correct']} sycophantic={final_eval['sycophantic']} "
                f"other={final_eval['other']} total={final_eval['total']})"
            )

        return {
            "best_suffix_str": best_suffix_str,
            "best_suffix_ids": best_suffix_ids.cpu().tolist()
            if best_suffix_ids is not None
            else None,
            "best_discrete_loss": best_discrete_loss,
            "relaxed_losses": all_losses,
            "discrete_losses": all_discrete_losses,
            "kl_losses": all_kl_losses,
            "final_eval": final_eval,
        }
