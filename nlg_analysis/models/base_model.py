from typing import List


class BaseModel:
    @staticmethod
    def approach() -> str:
        raise NotImplementedError

    def generate_transcript(self, questions: List):
        raise NotImplementedError
