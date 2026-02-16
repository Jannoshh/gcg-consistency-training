# You can't reduce sycophancy with discrete GCG suffixes from PGD

Negative result: Projected Gradient Descent (PGD) suffix optimization cannot produce discrete adversarial suffixes that reduce sycophancy in Gemma 2 2B. The continuous relaxation works (CE loss drops to ~1.0), but argmax discretization destroys the solution with a consistent 90%+ relaxation gap.

This was an MVP to evaluate whether discrete suffix optimization could reduce evaluation awareness in LLMs. The intended application is black-box optimization (GCG-style) on evaluation awareness, but we first tested PGD as an upper bound on discrete suffix performance (since PGD has gradient access and is faster than GCG). If PGD can't produce working discrete suffixes, GCG won't either.

We tested on sycophancy as a proxy for evaluation awareness, since both involve the model changing behavior based on contextual cues.

## Background

[Bias-augmented Consistency Training (BCT)](https://arxiv.org/abs/2308.09138) reduces sycophancy by finetuning on (biased prompt, correct response) pairs, where correct responses are generated from clean (unbiased) prompts. We attempted to achieve the same effect at inference time using an adversarial suffix instead of finetuning, optimized via [PGD from Geisler et al. 2024](https://arxiv.org/abs/2402.09154).

The idea: append a universal suffix to biased MCQ prompts so the model produces the same chain-of-thought response it would give on the clean (unbiased) version of the prompt.

## Method

1. **Generate reference responses**: Run the model on clean (unbiased) prompts to get correct CoT responses
2. **PGD optimization**: Optimize a continuous relaxation over `[suffix_len, vocab_size]` on the probability simplex, minimizing CE loss between model output on `(biased_prompt + suffix)` and the reference response
3. **Entropy annealing**: Tsallis q=2 entropy projection is annealed from 0 to 0.4 to gradually force one-hot (discrete) distributions
4. **Discretization**: `argmax` per position converts continuous factors to discrete tokens

## Results

| Configuration | Relaxed Loss | Best Discrete Loss | Gap | Sycophancy Rate |
|---|---|---|---|---|
| 25-token suffix, 2000 steps | ~2-3 | 6.25 | ~70% | No improvement |
| 100-token suffix, 2000 steps | ~1.0 | 7.56 | ~94% | No improvement |

The soft prompt (continuous embedding vectors) reduces CE loss effectively, proving the model *can* be steered toward correct responses with input perturbations. But discretizing those continuous vectors into actual tokens via argmax produces gibberish that doesn't transfer the steering effect.

The relaxation gap *increases* with more suffix tokens (70% -> 94%), suggesting that longer suffixes give PGD more room to exploit the continuous space in ways that have no discrete equivalent.

### Training dynamics (100-token suffix)

| Step | Relaxed Loss | Discrete Loss | Gap |
|---|---|---|---|
| 0 | 15.38 | 17.08 | 10% |
| 50 | 1.69 | 19.14 | 91% |
| 100 | 1.38 | 15.62 | 91% |
| 200 | 1.55 | 18.36 | 92% |
| 300 | 1.06 | 16.59 | 94% |

Entropy annealing (designed to close the gap) makes things worse: the best discrete loss is found early when entropy factor is low, before the projection forces distributions toward one-hot.

## Why it doesn't work

The core problem is that sycophancy correction requires changing the model's chain-of-thought reasoning, not just flipping a single output token. The CE target is a full CoT response (e.g. "Let me analyze this step by step... The answer is B. <answer>B</answer>"). Matching this requires coordinated changes across many suffix positions, and the argmax discretization of each position independently cannot preserve the subtle inter-position dependencies that the continuous solution exploits.

This contrasts with typical GCG/PGD attacks where the target is a short affirmative prefix ("Sure, here is...") that can be triggered by a few key tokens.

## Implications for evaluation awareness

Since PGD (with full gradient access) cannot produce discrete suffixes that reduce sycophancy, black-box methods like GCG will not work either. This rules out discrete suffix optimization as a viable approach for reducing evaluation awareness at inference time — at least for targets that require changing the model's reasoning process rather than a short output prefix.

## Replication

### Prerequisites

- Python 3.10+
- [Modal](https://modal.com) account with GPU access
- Modal secrets configured: `huggingface`, `wandb`
- Training data: sycophancy JSONL with fields `{clean_prompt, wrapped_prompt, correct_answer, suggested_wrong}`

### Setup

```bash
pip install -e .
modal setup  # configure Modal credentials
```

### Run PGD training

```bash
# Default: 100-token suffix, 2000 steps, 500-step entropy anneal
modal run --detach src/gcg_eval/modal_app.py --limit 500

# Customize
modal run --detach src/gcg_eval/modal_app.py \
    --suffix-length 100 \
    --num-steps 2000 \
    --entropy-anneal-steps 500 \
    --batch-size 8 \
    --limit 500
```

### Evaluate a suffix

```bash
modal run src/gcg_eval/modal_app.py \
    --mode pgd-eval \
    --pgd-suffix "your suffix string" \
    --limit 50
```

### Key things to look at

- `pgd/relaxed_loss` vs `pgd/discrete_loss` on wandb — the gap between these is the core finding
- `pgd/relaxation_gap` — should be >0.9 consistently, confirming discretization is the bottleneck
- Final eval `sycophancy_rate` — should be unchanged from baseline (~50%)

## References

- Geisler et al. (2024). [Attacking Large Language Models with Projected Gradient Descent](https://arxiv.org/abs/2402.09154)
- Wei & Zou (2023). [Simple Synthetic Data Reduces Sycophancy in Large Language Models](https://arxiv.org/abs/2308.09138)
- Zou et al. (2023). [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
