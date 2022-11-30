# Code based on ELIZA
# Copyright (C) 2019 Szymon Jessa
import json
import re
from typing import List

import spacy

from nlg_analysis.models.base_model import BaseModel


class RuleModel(BaseModel):
    """Rule based model."""

    def __init__(
        self,
        path2script: str = "data/example_rule_model_script.json",
        spacy_model: str = "pl_core_news_lg",
    ):
        self.memstack = []
        self.substitutions = {}
        self.script_memory = {}
        try:
            with open(path2script, "r") as f:
                self.script = json.load(f)
        except FileNotFoundError:
            print(f"File {path2script} not found.")
            self.script = {}
        self.lang_model = spacy.load(spacy_model)

    def generate_transcript(self, questions: List) -> str:
        """Generate conversation transcript."""
        trans_txt = ""
        for q in questions:
            trans_txt = trans_txt + q + "\n"
            answ = self.process(self.lang_model(q))
            trans_txt = trans_txt + answ + "\n"
        return trans_txt

    @staticmethod
    def approach() -> str:
        """Return the name of implemented approach."""
        return "RULE_MODEL"

    def process(self, user_input):
        keystack = self.get_keystack(user_input)
        user_input_trans = " ".join(map(lambda w: w.lemma_, user_input))
        resp = ""
        for kw in keystack:
            rule = self.script[kw]
            if re.search(rule["decomposition"], user_input.text.lower()):
                trans = rule["reassembly"].pop(0)
                rule["reassembly"].append(trans)
                resp = re.sub(
                    rule["decomposition"],
                    trans,
                    user_input.text.lower(),
                    count=1,
                )
                break
        if resp == "":
            if self.memstack:
                resp = self.memstack.pop(0)
            else:
                resp = self.script["none"]["reassembly"][0]
        self.memorize_user_input(user_input, user_input_trans)
        return str(resp)

    def get_keystack(self, user_input):
        keystack = []
        for token in user_input:
            if token.lemma_ in self.script:
                keystack.append(
                    (token.lemma_, self.script[token.lemma_].get("rank", 0))
                )
        keystack = sorted(keystack, key=lambda i: i[1], reverse=True)
        keystack = [w for w, r in keystack]
        return keystack

    def memorize_user_input(self, user_input, user_input_trans):
        memory_keywords = []
        for token in user_input:
            if token.lemma_ in self.script_memory:
                memory_keywords.append(token.lemma_)
        memory_keywords = list(set(memory_keywords))

        for k in memory_keywords:
            memresp = re.sub(
                self.script_memory[k]["decomposition"],
                self.script_memory[k]["reassembly"][0],
                user_input_trans,
            )
            self.memstack.append(memresp)
