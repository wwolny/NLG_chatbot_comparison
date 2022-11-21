from nlg_analysis.models.base_model import BaseModel


class GPTModel(BaseModel):
    def generate_transcript(self):
        pass

    @staticmethod
    def approach() -> str:
        return "GPT_MODEL"

