from nlg_analysis.models.base_model import BaseModel


class RNNModel(BaseModel):
    @staticmethod
    def approach() -> str:
        return "RNN_MODEL"

