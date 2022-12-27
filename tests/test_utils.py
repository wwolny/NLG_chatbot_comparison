import pandas as pd

from nlg_analysis.utils import (
    load_questions,
    parse_arguments,
    save_conversations,
)


def test_parse_args_short():
    path_to_file = "wwolny/config_file.yaml"
    new_args = ["-cf", path_to_file]
    parser = parse_arguments(new_args)
    assert parser.config_file == path_to_file


def test_parse_args_long():
    path_to_file = "wwolny/config_file.yaml"
    new_args = ["--config-file", path_to_file]
    parser = parse_arguments(new_args)
    assert parser.config_file == path_to_file


def test_load_questions():
    path_to_file = "tests/resources/sample_questions.csv"
    df_q = load_questions(path_to_file)
    assert df_q.shape == (2, 8)


def test_load_questions_header():
    path_to_file = "tests/resources/sample_questions.csv"
    df_q = load_questions(path_to_file, header=None)
    assert df_q.shape == (3, 8)


def test_save_conversations():
    path_to_file = "tests/resources/saved_conversations.txt"
    conv_1 = "Hej!\nCześć!"
    conversations = [{"ql_id": 0, "conversations": conv_1, "model": "test"}]
    save_conversations(path_to_file, conversations)
    df_conversations = pd.read_csv(path_to_file)
    assert df_conversations.shape == (1, 4)
