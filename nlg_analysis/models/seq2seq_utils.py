import random
import re
import unicodedata
from typing import List, Tuple

import torch

from nlg_analysis.models.conversation_side import ConversationSide


def unicodeToAscii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalizeString(s: str) -> str:
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZęóąśłżźćńĘÓĄŁŻŹĆŃ.!?]+", r" ", s)
    return s


def readConvSides(
    questions_path: str, answers_path: str
) -> Tuple[ConversationSide, ConversationSide, List[Tuple[str, str]]]:
    print("Reading lines...")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = f.read().split("\n")
    with open(answers_path, "r", encoding="utf-8") as f:
        answers = f.read().split("\n")
    questions = questions[:-2]
    answers = answers[:-2]
    pairs = list(zip(questions, answers))
    random.shuffle(pairs)

    input_side = ConversationSide()
    output_side = ConversationSide()

    return input_side, output_side, pairs


def indexesFromSentence(
    conv_side: ConversationSide, sentence: str
) -> List[int]:
    return [
        conv_side.word2index[word]
        for word in sentence.split(" ")
        if word in conv_side.word2index.keys()
    ]


def tensorFromSentence(
    conv_side: ConversationSide,
    sentence: str,
    eos_token: int,
    device: torch.device,
) -> torch.Tensor:
    indexes = indexesFromSentence(conv_side, sentence)
    indexes.append(eos_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(
    pair: Tuple[str, str],
    input_side: ConversationSide,
    output_side: ConversationSide,
    eos_token: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_tensor = tensorFromSentence(input_side, pair[0], eos_token, device)
    target_tensor = tensorFromSentence(output_side, pair[1], eos_token, device)
    return input_tensor, target_tensor


def prepareData(
    questions_path: str, answers_path: str
) -> Tuple[ConversationSide, ConversationSide, List[Tuple[str, str]]]:
    input_side, output_side, pairs = readConvSides(
        questions_path, answers_path
    )
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_side.addSentence(pair[0])
        output_side.addSentence(pair[1])
    return input_side, output_side, pairs
