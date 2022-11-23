from typing import List

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models.base_model import BaseModel


class RNNModel(BaseModel):
    def generate_transcript(self, questions: List) -> str:
        pass

    @staticmethod
    def approach() -> str:
        return "RNN_MODEL"

    def train(self, output_path: str, train_config: TrainConfig):
        pass
