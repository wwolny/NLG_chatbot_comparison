from nlg_analysis.models.base_model import BaseModel


class BERTModel(BaseModel):
    @staticmethod
    def approach() -> str:
        return "BERT_MODEL"

