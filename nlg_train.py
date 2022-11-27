import logging
import os
import sys

import yaml

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models import BERTModel, GPTModel, RNNModel
from nlg_analysis.utils import parse_arguments

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
    train_cfg = TrainConfig.from_dict(config)

    # Setup models
    models = []
    if train_cfg.gpt_model:
        models.append(
            {
                "model": GPTModel(model_name=train_cfg.gpt_model_name),
                "output_path": train_cfg.gpt_output_path,
            }
        )
    if train_cfg.rnn_model:
        models.append(
            {"model": RNNModel(), "output_path": train_cfg.rnn_output_path}
        )
    if train_cfg.bert_model:
        models.append(
            {
                "model": BERTModel(model_name=train_cfg.bert_model_name),
                "output_path": train_cfg.bert_output_path,
            }
        )
    logger.info(
        "Train models: {0}".format(
            ", ".join([model["model"].approach() for model in models])
        )
    )

    # Train models
    for model in models:
        logger.info("Train model {0}".format(model["model"].approach()))
        model["model"].train(model["output_path"], train_cfg)


if __name__ == "__main__":
    main()
