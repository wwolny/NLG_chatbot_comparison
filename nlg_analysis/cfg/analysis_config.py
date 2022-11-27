import datetime
from dataclasses import dataclass

from nlg_analysis.cfg.base_config import BaseConfig


@dataclass
class AnalysisConfig(BaseConfig):
    ts: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file: str = "output/output_analysis_{0}.csv".format(ts)
    question_file: str = "question.csv"
    gpt_model_path: str = "model_ckpt/gpt_.ckpt"
    bert_model_path: str = "model_ckpt/bert_.ckpt"
    rnn_model_path: str = "model_ckpt/rnn_.ckpt"
