# You can't reduce sycophancy with discrete GCG suffixes from PGD

Negative result: [PGD](https://arxiv.org/abs/2402.09154) can optimize continuous suffix embeddings that steer Gemma 2 2B toward correct answers on biased MCQ prompts (CE loss ~1.0). But converting those embeddings to discrete tokens destroys the effect — sycophancy rate is unchanged from baseline.

This was an MVP for discrete suffix optimization against evaluation awareness. We tested PGD as an upper bound on what [GCG](https://arxiv.org/abs/2307.15043) could achieve (PGD has gradient access and is faster). If PGD can't produce working discrete suffixes, GCG won't either. We used sycophancy as a proxy, since both involve the model changing behavior based on contextual cues.

## Method

Following [BCT](https://arxiv.org/abs/2308.09138), we generate correct chain-of-thought responses on clean (unbiased) prompts, then optimize a suffix so the model reproduces those responses when given the biased version of the prompt.

PGD maintains a continuous distribution over the vocabulary for each suffix position. During training, these soft embeddings are gradually pushed toward discrete (one-hot) distributions via entropy annealing. The final suffix is obtained by taking the argmax at each position.

## Results

| Configuration | Soft Loss | Discrete Loss | Sycophancy Rate |
|---|---|---|---|
| 25-token suffix, 2000 steps | ~2-3 | 6.25 | No improvement |
| 100-token suffix, 1300 steps | 0.69 | 4.75 (best) | No improvement |

The soft embeddings work — they reduce the loss to ~0.7, meaning the model with continuous suffix vectors largely produces the correct response. But the discretized tokens don't transfer this effect.

### Progressive discretization

We also tried freezing suffix positions one at a time (most confident first) instead of argmax-ing all at once, hoping the remaining positions could adapt. This didn't help — the best discrete loss was 4.40 (100 tokens) and 6.35 (40 tokens), both comparable to the all-at-once baseline.

| Configuration | Soft Loss | Best Discrete Loss |
|---|---|---|
| 100-token, progressive | 0.87 | 4.40 |
| 40-token, progressive | ~10-16 | 6.35 |

As positions freeze, the relaxed loss climbs because the optimizer runs out of free positions to compensate for locked-in bad tokens.

## Replication

### Prerequisites

- Python 3.10+
- [Modal](https://modal.com) account with GPU access
- Modal secrets: `huggingface`, `wandb`
- Training data: sycophancy JSONL with fields `{clean_prompt, wrapped_prompt, correct_answer, suggested_wrong}`

### Setup

```bash
pip install -e .
modal setup
```

### Run PGD training

```bash
modal run --detach src/gcg_consistency_training/modal_app.py --limit 500

modal run --detach src/gcg_consistency_training/modal_app.py \
    --suffix-length 100 \
    --num-steps 2000 \
    --entropy-anneal-steps 500 \
    --batch-size 8 \
    --limit 500
```

### Evaluate a suffix

```bash
modal run src/gcg_consistency_training/modal_app.py \
    --mode pgd-eval \
    --pgd-suffix "your suffix string" \
    --limit 50
```

### Key wandb metrics

- `pgd/relaxed_loss` vs `pgd/discrete_loss` — the gap is the core finding
- Final eval `sycophancy_rate` — unchanged from baseline (~50%)

## References

- Geisler et al. (2024). [Attacking Large Language Models with Projected Gradient Descent](https://arxiv.org/abs/2402.09154)
- Wei & Zou (2023). [Simple Synthetic Data Reduces Sycophancy in Large Language Models](https://arxiv.org/abs/2308.09138)
- Zou et al. (2023). [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
