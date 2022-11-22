import argparse
from typing import List, Dict

import pandas as pd


def load_questions(filename: str):
    return pd.read_csv(filename, sep=";", header=0, quotechar="'")


def save_conversations(filename: str, conversations: List[Dict]):
    pd.DataFrame(conversations).to_csv(filename)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-cf")
    return parser.parse_args(args)
