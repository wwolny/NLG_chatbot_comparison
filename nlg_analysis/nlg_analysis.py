import sys

from nlg_analysis.models import RuleModel, RNNModel, BERTModel, GPTModel
from nlg_analysis.utils import parse_arguments, load_questions, save_conversations


def main(question_file: str, output_file: str):
    question_line = load_questions(question_file)

    models = [RuleModel(), RNNModel(), BERTModel(), GPTModel()]
    conversations = []
    for model in models:
        for ql_id, questions in enumerate(question_line):
            conversation = model.generate_transcript(questions)
            conversations.append({"ql_id": ql_id, "conversation": conversation, "model": model.approach()})
    save_conversations(output_file, conversations)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(question_file=args.question_file, output_file=args.output_file)
