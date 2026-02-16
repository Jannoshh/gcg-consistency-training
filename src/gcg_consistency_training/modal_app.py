"""Modal deployment for PGD suffix optimization.

Optimizes a discrete adversarial suffix via projected gradient descent
(Geisler et al. 2024) to reduce sycophancy in Gemma 2 2B.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("gcg-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "accelerate>=0.27.0",
        "pydantic>=2.0.0",
        "wandb>=0.16.0",
    )
    .add_local_python_source("gcg_consistency_training")
)

results_volume = modal.Volume.from_name("gcg-eval-results", create_if_missing=True)
model_volume = modal.Volume.from_name("gcg-eval-models", create_if_missing=True)

GPU = "A100-80GB"
TIMEOUT = 14400  # 4 hours
secrets = [
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("wandb"),
]


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    secrets=secrets,
    volumes={
        "/results": results_volume,
        "/models": model_volume,
    },
)
def run_pgd_eval(
    eval_jsonl: str,
    suffix: str = "",
    max_samples: int = 50,
) -> dict:
    """Evaluate sycophancy rate with and without a suffix."""
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gcg_consistency_training.config import Config
    from gcg_consistency_training.pgd_suffix import PGDSuffixOptimizer

    config = Config()

    print(f"Loading model: {config.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id, cache_dir="/models"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
    )

    eval_data = [
        json.loads(line)
        for line in eval_jsonl.strip().split("\n")
        if line.strip()
    ]
    print(f"Loaded {len(eval_data)} eval samples, using {max_samples}")

    optimizer = PGDSuffixOptimizer(model, tokenizer, config)

    # Baseline: no suffix
    print("\n=== Baseline (no suffix) ===")
    dummy_ids = torch.zeros(0, dtype=torch.long, device=optimizer.device)
    baseline = optimizer.evaluate_accuracy(eval_data, dummy_ids, max_samples=max_samples)
    print(f"Baseline: {baseline}")

    # With suffix
    result_with_suffix = None
    if suffix:
        print(f"\n=== With suffix: {suffix[:80]}... ===")
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False, return_tensors="pt")[0].to(optimizer.device)
        print(f"Suffix tokens: {len(suffix_ids)}")
        result_with_suffix = optimizer.evaluate_accuracy(eval_data, suffix_ids, max_samples=max_samples)
        print(f"With suffix: {result_with_suffix}")

    return {
        "baseline": baseline,
        "with_suffix": result_with_suffix,
        "suffix": suffix,
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    secrets=secrets,
    volumes={
        "/results": results_volume,
        "/models": model_volume,
    },
)
def run_pgd_training(
    train_jsonl: str,
    eval_jsonl: str = "",
    num_steps: int = 5000,
    suffix_length: int = 25,
    batch_size: int = 8,
    lr: float = 0.1,
    limit: int = 0,
    eval_every: int = 100,
    kl_weight: float = 0.0,
    entropy_anneal_steps: int = 250,
    progressive_discretize: bool = False,
    discretize_warmup_steps: int = 500,
) -> dict:
    """Run PGD suffix optimization on Modal GPU."""
    import json
    import datetime
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gcg_consistency_training.config import Config
    from gcg_consistency_training.pgd_suffix import PGDSuffixOptimizer

    config = Config()
    config.pgd.num_steps = num_steps
    config.pgd.suffix_length = suffix_length
    config.pgd.batch_size = batch_size
    config.pgd.lr = lr
    config.pgd.eval_every = eval_every
    config.pgd.kl_weight = kl_weight
    config.pgd.entropy_anneal_steps = entropy_anneal_steps
    config.pgd.progressive_discretize = progressive_discretize
    config.pgd.discretize_warmup_steps = discretize_warmup_steps

    wandb.init(
        entity="janneselstner",
        project="eval-awareness-gcg",
        config=config.model_dump(),
        tags=["pgd", config.model.model_id.split("/")[-1]],
    )

    print(f"Loading model: {config.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id, cache_dir="/models"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
    )

    train_data = [
        json.loads(line)
        for line in train_jsonl.strip().split("\n")
        if line.strip()
    ]
    if limit > 0:
        train_data = train_data[:limit]
    print(f"Loaded {len(train_data)} training samples")

    eval_data = None
    if eval_jsonl:
        eval_data = [
            json.loads(line)
            for line in eval_jsonl.strip().split("\n")
            if line.strip()
        ]
        print(f"Loaded {len(eval_data)} eval samples")

    optimizer = PGDSuffixOptimizer(model, tokenizer, config)
    result = optimizer.train(train_data, eval_data, wandb_run=wandb)

    # Save result
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = f"/results/pgd_{ts}.json"
    save_data = {
        "best_suffix_str": result["best_suffix_str"],
        "best_suffix_ids": result["best_suffix_ids"],
        "best_discrete_loss": result["best_discrete_loss"],
        "final_eval": result["final_eval"],
        "config": config.model_dump(),
    }
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    results_volume.commit()
    print(f"Saved to {result_path}")

    artifact = wandb.Artifact(f"pgd-result-{ts}", type="result")
    artifact.add_file(result_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    return save_data


@app.local_entrypoint()
def main(
    mode: str = "pgd",
    train_data_path: str = "data/sycophancy_train.jsonl",
    eval_data_path: str = "data/sycophancy_eval.jsonl",
    num_steps: int = 2000,
    suffix_length: int = 100,
    batch_size: int = 8,
    lr: float = 0.1,
    limit: int = 0,
    eval_every: int = 0,
    kl_weight: float = 0.0,
    pgd_suffix: str = "",
    entropy_anneal_steps: int = 500,
    progressive_discretize: bool = False,
    discretize_warmup_steps: int = 500,
):
    """Entry point for `modal run`.

    Usage:
        modal run --detach src/gcg_consistency_training/modal_app.py --num-steps 2000 --suffix-length 100 --limit 500
        modal run --detach src/gcg_consistency_training/modal_app.py --mode pgd-eval --pgd-suffix "your suffix here" --limit 50
    """
    import json

    if mode == "pgd-eval":
        eval_path = Path(eval_data_path)
        eval_jsonl = eval_path.read_text() if eval_path.exists() else Path(train_data_path).read_text()

        result = run_pgd_eval.remote(
            eval_jsonl=eval_jsonl,
            suffix=pgd_suffix,
            max_samples=limit if limit > 0 else 50,
        )

        print(f"\nBaseline: {json.dumps(result['baseline'], indent=2)}")
        if result['with_suffix']:
            print(f"\nWith suffix: {json.dumps(result['with_suffix'], indent=2)}")

    elif mode == "pgd":
        train_jsonl = Path(train_data_path).read_text()
        eval_jsonl = ""
        eval_path = Path(eval_data_path)
        if eval_path.exists():
            eval_jsonl = eval_path.read_text()

        result = run_pgd_training.remote(
            train_jsonl=train_jsonl,
            eval_jsonl=eval_jsonl,
            num_steps=num_steps,
            suffix_length=suffix_length,
            batch_size=batch_size,
            lr=lr,
            limit=limit,
            eval_every=eval_every,
            kl_weight=kl_weight,
            entropy_anneal_steps=entropy_anneal_steps,
            progressive_discretize=progressive_discretize,
            discretize_warmup_steps=discretize_warmup_steps,
        )

        print(f"\nBest suffix: {result['best_suffix_str']}")
        print(f"Best discrete loss: {result['best_discrete_loss']:.4f}")
        if result.get("final_eval"):
            print(f"Final eval: {json.dumps(result['final_eval'], indent=2)}")
    else:
        print(f"Unknown mode: {mode}. Use 'pgd' or 'pgd-eval'.")
