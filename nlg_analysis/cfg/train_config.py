import datetime
from dataclasses import dataclass

from nlg_analysis.cfg.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    # Dataset paths
    train_ds_path: str = "data/datasets/"
    test_ds_path: str = "data/datasets/"

    # Run models
    gpt_model: bool = True
    rnn_model: bool = True
    bert_model: bool = True

    # Models information
    gpt_model_name: str = "flax-community/papuGaPT2"
    bert_model_name: str = "dkleczek/bert-base-polish-uncased-v1"
    ts: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gpt_output_path: str = "model_ckpt/gpt_{0}.ckpt".format(ts)
    bert_output_path: str = "model_ckpt/bert_{0}.ckpt".format(ts)
    rnn_output_path: str = "model_ckpt/rnn_{0}.ckpt".format(ts)

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

    # RNN parameters
    seq_length: int = 100
    buffer_size = 64
    batch_size = 10000
