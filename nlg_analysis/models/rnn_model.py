from nlg_analysis.models.base_model import BaseModel


class RNNModel(BaseModel):
    def generate_transcript(self):
        pass

    @staticmethod
    def approach() -> str:
        return "RNN_MODEL"

