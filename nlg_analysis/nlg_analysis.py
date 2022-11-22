import os
import sys
import yaml
import logging

from nlg_analysis.cfg import AnalysisConfig
from nlg_analysis.models import RuleModel, RNNModel, BERTModel, GPTModel
from nlg_analysis.utils import parse_arguments, load_questions, save_conversations

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=os.environ.get("LOGLEVEL", "INFO"),
)


def main():
    args = parse_arguments(sys.argv[1:])

    # Load config file
    logger.info("Loading config...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    analysis_cfg = AnalysisConfig.from_dict(config)

    question_line = load_questions(analysis_cfg.question_file)

    models = [RuleModel(), RNNModel(), BERTModel(), GPTModel()]
    conversations = []
    for model in models:
        for ql_id, questions in enumerate(question_line):
            conversation = model.generate_transcript(questions)
            conversations.append({"ql_id": ql_id, "conversation": conversation, "model": model.approach()})
    save_conversations(analysis_cfg.output_file, conversations)


if __name__ == "__main__":
    main()
