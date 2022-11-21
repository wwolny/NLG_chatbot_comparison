from nlg_analysis.models.base_model import BaseModel


class GPTModel(BaseModel):
    @staticmethod
    def approach() -> str:
        return "GPT_MODEL"

