import datetime
from dataclasses import dataclass, fields


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, obj: dict):
        fields_names = [fld.name for fld in fields(cls)]
        dct = {k: v for (k, v) in obj.items() if k in fields_names}
        return cls(**dct)


@dataclass
class AnalysisConfig(BaseConfig):
    ts: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file: str = 'output/output_analysis_{0}.csv'.format(ts)
    question_file: str = 'question.csv'
    gpt_model_path: str = 'model_ckpt/gpt_.ckpt'
    bert_model_path: str = 'model_ckpt/bert_.ckpt'
    rnn_model_path: str = 'model_ckpt/rnn_.ckpt'


@dataclass
class TrainConfig(BaseConfig):
    gpt_model: bool = True
    rnn_model: bool = True
    bert_model: bool = True
    ts: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gpt_output_path: str = "model_ckpt/gpt_{0}.ckpt".format(ts)
    bert_output_path: str = "model_ckpt/bert_{0}.ckpt".format(ts)
    rnn_output_path: str = "model_ckpt/rnn_{0}.ckpt".format(ts)
