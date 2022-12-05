import argparse
from typing import Dict, List

import pandas as pd


def load_questions(
    filename: str, sep: str = ";", header: int = 0, quotechar: str = "'"
):
    """Load questions from given csv file."""
    return pd.read_csv(filename, sep=sep, header=header, quotechar=quotechar)


def save_conversations(filename: str, conversations: List[Dict]):
    """Save transcript of conversation to csv file."""
    pd.DataFrame(conversations).to_csv(filename)


def parse_arguments(args):
    """Parse script arguments into parser object."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-cf")
    return parser.parse_args(args)
