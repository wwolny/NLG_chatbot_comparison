class BaseModel:
    @staticmethod
    def approach() -> str:
        raise NotImplementedError

    def generate_transcript(self):
        raise NotImplementedError
