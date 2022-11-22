from typing import List

from nlg_analysis.models.base_model import BaseModel


class GPTModel(BaseModel):
    def generate_transcript(self, questions: List):
        pass

    @staticmethod
    def approach() -> str:
        return "GPT_MODEL"

    def train(self, output_path):
        pass

