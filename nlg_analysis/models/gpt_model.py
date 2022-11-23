import pathlib
from typing import List
from transformers import pipeline, Trainer, TrainingArguments, AutoModelWithLMHead, TextDataset, \
    DataCollatorForLanguageModeling, AutoTokenizer

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models.base_model import BaseModel


class GPTModel(BaseModel):
    def __init__(
            self,
            path2model: str = None,
            gpt_model: str = "flax-community/papuGaPT2",
    ):
        if path2model is not None:
            config_path: str = path2model + "config.json"
            if pathlib.Path(config_path).exists():
                self.model = pipeline(
                    'text-generation',
                    model=path2model,
                    tokenizer=gpt_model,
                    config=config_path
                )
            else:
                raise FileNotFoundError
        else:
            self.model = AutoModelWithLMHead.from_pretrained(gpt_model)
            self.tokenizer = AutoTokenizer.from_pretrained(gpt_model)

    def generate_transcript(self, questions: List) -> str:
        output_txt = ""
        for q in questions:
            full_quest = "<q> " + q + "\n<a>"
            answers = self.model(full_quest, max_length=40)[0]["generated_text"]
            answers = answers[len(full_quest):]
            answers = answers.split("\n")[0]
            output_txt += q + "\n" + answers.strip() + "\n"
        return output_txt

    @staticmethod
    def approach() -> str:
        return "GPT_MODEL"

    def train(self, output_path: str, train_config: TrainConfig):
        train_dataset, test_dataset, data_collator = self.load_dataset(
            train_config.train_ds_path,
            train_config.test_ds_path
        )

        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=25,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_steps=1000,
            save_total_limit=3,
            save_steps=10000,
            warmup_steps=600,
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()

    def load_dataset(self, train_path: str, test_path: str):
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=64)

        test_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=test_path,
            block_size=64)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False,
        )
        return train_dataset, test_dataset, data_collator
