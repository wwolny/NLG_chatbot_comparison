import argparse
from typing import List, Dict

import pandas as pd


def load_questions(filename: str):
    return pd.read_csv(filename, sep=";", header=0, )


def save_conversations(filename: str, conversations: List[Dict]):
    df_convs = pd.DataFrame(conversations)
    df_convs.to_csv(filename)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    return parser.parse_args(args)
