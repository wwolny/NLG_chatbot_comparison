from typing import List

from nlg_analysis.models.base_model import BaseModel


class RuleModel(BaseModel):
    """Rule based model."""

    def generate_transcript(self, questions: List) -> str:
        """Generate conversation transcript."""
        pass

    @staticmethod
    def approach() -> str:
        """Return the name of implemented approach."""
        return "RULE_MODEL"
