import datetime
from dataclasses import dataclass

from nlg_analysis.cfg.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    gpt_model: bool = True
    rnn_model: bool = True
    bert_model: bool = True

    # Mdels paths
    ts: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gpt_output_path: str = "model_ckpt/gpt_{0}.ckpt".format(ts)
    bert_output_path: str = "model_ckpt/bert_{0}.ckpt".format(ts)
    rnn_output_path: str = "model_ckpt/rnn_{0}.ckpt".format(ts)
    train_ds_path: str = "data/datasets/train"
    test_ds_path: str = "data/datasets/test"

    # Common training parameters
    train_batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 25
    eval_step: int = 1000
    save_steps: int = 10000

    # GPT-2 parameters
    save_total_limit: int = 3
    warmup_steps: int = 600
    predictions_loss_only: bool = True
    block_size: int = 64

    # BERT parameters
    weight_decay: float = 0.01
    bert_load_best_model: bool = True
    logging_steps: int = 200
