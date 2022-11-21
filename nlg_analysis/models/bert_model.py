from nlg_analysis.models.base_model import BaseModel


class BERTModel(BaseModel):
    def generate_transcript(self):
        pass

    @staticmethod
    def approach() -> str:
        return "BERT_MODEL"

