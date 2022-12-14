import logging
import os
import sys

import yaml

from nlg_analysis.cfg import AnalysisConfig
from nlg_analysis.models import BERTModel, GPTModel, RNNModel, RuleModel
from nlg_analysis.utils import (
    load_questions,
    parse_arguments,
    save_conversations,
)

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=os.environ.get("LOGLEVEL", "INFO"),
)


def main():
    args = parse_arguments(sys.argv[1:])

    # Load config file
    logger.info("Loading config...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    analysis_cfg = AnalysisConfig.from_dict(config)

    # Load questions
    logger.info("Loading questions...")
    df_ql = load_questions(analysis_cfg.question_file)

    logger.info("Init models...")
    models = [
        RuleModel(analysis_cfg.rule_model_path, analysis_cfg.spacy_model),
        RNNModel(),
        BERTModel(),
        GPTModel(),
    ]

    logger.info("Run conversations...")
    conversations = []
    for model in models:
        logger.info("Run model {0}".format(model.approach()))
        for ql_id, questions in df_ql.iterrows():
            conversation = model.generate_transcript(questions.tolist())
            conversations.append(
                {
                    "ql_id": ql_id,
                    "conversation": conversation,
                    "model": model.approach(),
                }
            )
    save_conversations(analysis_cfg.output_file, conversations)


if __name__ == "__main__":
    main()
