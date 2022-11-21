from nlg_analysis.models.base_model import BaseModel


class RuleModel(BaseModel):
    @staticmethod
    def approach() -> str:
        return "RULE_MODEL"
