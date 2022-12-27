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
