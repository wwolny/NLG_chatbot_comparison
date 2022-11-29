from typing import List


class BaseModel:
    """Model class template."""

    @staticmethod
    def approach() -> str:
        """Return the name of implemented approach."""
        raise NotImplementedError

    def generate_transcript(self, questions: List) -> str:
        """Generate conversation transcript."""
        raise NotImplementedError
