from typing import List

import datasets
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models.base_model import BaseModel


class BERTModel(BaseModel):

    def __init__(
            self,
            path2model: str = None,
            model_name: str = "dkleczek/bert-base-polish-uncased-v1",
    ):
        if path2model is not None:
            self.model = BertForSequenceClassification.from_pretrained(path2model)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=88)
            self.encoder_max_length = 64
            self.lab_key = pd.read_csv("data/label-key.txt", sep=";")
            self.lab_key = self.lab_key.set_index("label").to_dict()

    def generate_transcript(self, questions: List) -> str:
        full_txt = ""
        for q in questions:
            pred_q = self.get_prediction(q)
            pred_l = pred_q["logits"].argmax().item()
            ans = self.lab_key["answer"][pred_l]
            full_txt += q + '\n' + ans + '\n'
        return full_txt

    @staticmethod
    def approach() -> str:
        return "BERT_MODEL"

    def train(self, output_path: str, train_config: TrainConfig):
        dataset_train = datasets.load_dataset('csv', data_files=train_config.train_ds_path, split=datasets.Split.TRAIN)
        dataset_val = datasets.load_dataset('csv', data_files=train_config.test_ds_path, split=datasets.Split.TRAIN)
        batch_size = 4

        train_data = dataset_train.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["input", "label", "Unnamed: 0"]
        )

        eval_data = dataset_val.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["input", "label", "Unnamed: 0"]
        )

        eval_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"],
        )

        training_args = TrainingArguments(
            num_train_epochs=10,
            per_device_train_batch_size=16,
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_steps=200,
            evaluation_strategy="steps",
            output_dir=output_path,
            eval_steps=400,
            save_steps=1000,
        )
        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=train_data,
                          eval_dataset=eval_data)

        trainer.train()

    def process_data_to_model_inputs(self, batch):
        inputs = self.tokenizer(batch["input"],
                                padding="max_length",
                                truncation=True,
                                max_length=self.encoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = batch["label"]
        return batch

    def get_prediction(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length,
            return_tensors="pt"
        )
        outputs = self.model(**inputs)
        return outputs
