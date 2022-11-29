from typing import List

import datasets
import pandas as pd
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models.base_model import BaseModel


class BERTModel(BaseModel):
    """Fine tuned BERT classifier."""

    def __init__(
        self,
        path2model: str = None,
        model_name: str = "dkleczek/bert-base-polish-uncased-v1",
        num_labels: int = 88,
        encoder_max_length: int = 64,
    ):
        self.encoder_max_length = encoder_max_length
        if path2model is not None:
            self.model = BertForSequenceClassification.from_pretrained(
                path2model
            )
            self.lab_key = pd.read_csv("data/label-key.txt", sep=";")
            self.lab_key = self.lab_key.set_index("label").to_dict()
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )

    def generate_transcript(self, questions: List) -> str:
        """Generate conversation transcript."""
        full_txt = ""
        for q in questions:
            pred_q = self.get_prediction(q)
            pred_l = pred_q["logits"].argmax().item()
            ans = self.lab_key["answer"][pred_l]
            full_txt += q + "\n" + ans + "\n"
        return full_txt

    @staticmethod
    def approach() -> str:
        """Return the name of implemented approach."""
        return "BERT_MODEL"

    def train(self, output_path: str, train_config: TrainConfig):
        """Train model and save to given output path."""
        dataset_train = datasets.load_dataset(
            "csv",
            data_files=train_config.train_ds_path,
            split=datasets.Split.TRAIN,
        )
        dataset_val = datasets.load_dataset(
            "csv",
            data_files=train_config.test_ds_path,
            split=datasets.Split.TRAIN,
        )

        train_data = dataset_train.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=train_config.train_batch_size,
            remove_columns=["input", "label", "Unnamed: 0"],
        )

        eval_data = dataset_val.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=train_config.eval_batch_size,
            remove_columns=["input", "label", "Unnamed: 0"],
        )

        eval_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        training_args = TrainingArguments(
            num_train_epochs=train_config.epochs,
            per_device_train_batch_size=train_config.train_batch_size,
            weight_decay=train_config.weight_decay,
            load_best_model_at_end=train_config.bert_load_best_model,
            logging_steps=train_config.logging_steps,
            evaluation_strategy="steps",
            output_dir=output_path,
            eval_steps=train_config.eval_step,
            save_steps=train_config.save_steps,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
        )

        trainer.train()

    def process_data_to_model_inputs(self, batch):
        """Fix batch format."""
        inputs = self.tokenizer(
            batch["input"],
            padding="max_length",
            truncation=True,
            max_length=self.encoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = batch["label"]
        return batch

    def get_prediction(self, text: str):
        """Predict and return answer to a query."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        return outputs
