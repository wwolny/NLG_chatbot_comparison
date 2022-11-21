import argparse
import pandas as pd


def load_questions(filename: str):
    pass


def save_conversations(filename: str, conversations):
    df_convs = pd.DataFrame(conversations)
    df_convs.to_csv(filename)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    return parser.parse_args(args)
