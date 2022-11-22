import os
import sys
import yaml
import logging

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.utils import parse_arguments

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
    train_cfg = TrainConfig.from_dict(config)


if __name__ == "__main__":
    main()
