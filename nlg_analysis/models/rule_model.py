from nlg_analysis.models.base_model import BaseModel


class RuleModel(BaseModel):
    def generate_transcript(self):
        pass

    @staticmethod
    def approach() -> str:
        return "RULE_MODEL"
