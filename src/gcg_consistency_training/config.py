from pydantic import BaseModel, Field


class PGDParams(BaseModel):
    """Parameters for PGD suffix optimization (Geisler et al. 2024).

    Optimizes a discrete suffix via projected gradient descent on continuous
    relaxation. Each suffix position is a distribution over vocab on the
    probability simplex. Tsallis entropy projection is annealed to force
    discrete solutions.
    """

    suffix_length: int = 100
    num_steps: int = 2000
    lr: float = 0.1
    grad_clip: float = 20.0
    entropy_factor_start: float = 0.0
    entropy_factor_end: float = 0.4
    entropy_anneal_steps: int = 500
    batch_size: int = 8
    allow_non_ascii: bool = False
    seed: int = 42
    eval_every: int = 0  # 0 = only at end
    kl_weight: float = 0.0  # 0 = disabled
    train_data_path: str = "data/sycophancy_train.jsonl"
    eval_data_path: str = "data/sycophancy_eval.jsonl"


class ModelParams(BaseModel):
    """Parameters for the target model."""

    model_id: str = "google/gemma-2-2b-it"
    max_new_tokens: int = 512
    max_target_tokens: int = 50
    temperature: float = 0.0
    do_sample: bool = False


class ModalParams(BaseModel):
    """Parameters for Modal deployment."""

    gpu: str = "A100-80GB"
    timeout: int = 7200
    volume_name: str = "gcg-eval-results"


class Config(BaseModel):
    """Top-level configuration."""

    pgd: PGDParams = Field(default_factory=PGDParams)
    model: ModelParams = Field(default_factory=ModelParams)
    modal: ModalParams = Field(default_factory=ModalParams)
