import pathlib
from typing import List

from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    Trainer,
    TrainingArguments,
    pipeline,
)

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models.base_model import BaseModel


class GPTModel(BaseModel):
    """Fine tuned GPT-2 model."""

    def __init__(
        self,
        path2model: str = None,
        model_name: str = "flax-community/papuGaPT2",
    ):
        if path2model is not None:
            config_path: str = path2model + "config.json"
            if pathlib.Path(config_path).exists():
                self.model = pipeline(
                    "text-generation",
                    model=path2model,
                    tokenizer=model_name,
                    config=config_path,
                )
            else:
                raise FileNotFoundError
        else:
            self.model = AutoModelWithLMHead.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_transcript(self, questions: List) -> str:
        """Generate conversation transcript."""
        output_txt = ""
        for q in questions:
            full_quest = "<q> " + q + "\n<a>"
            answers = self.model(full_quest, max_length=40)[0][
                "generated_text"
            ]
            answers = answers[len(full_quest) :]
            answers = answers.split("\n")[0]
            output_txt += q + "\n" + answers.strip() + "\n"
        return output_txt

    @staticmethod
    def approach() -> str:
        """Return the name of implemented approach."""
        return "GPT_MODEL"

    def train(self, output_path: str, train_config: TrainConfig):
        """Train model and save to given output path."""
        train_dataset, test_dataset, data_collator = self.load_dataset(
            train_config.train_ds_path,
            train_config.test_ds_path,
            block_size=train_config.block_size,
        )

        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=train_config.epochs,
            per_device_train_batch_size=train_config.train_batch_size,
            per_device_eval_batch_size=train_config.eval_batch_size,
            eval_steps=train_config.eval_step,
            save_total_limit=train_config.save_total_limit,
            save_steps=train_config.save_steps,
            warmup_steps=train_config.warmup_steps,
            prediction_loss_only=train_config.predictions_loss_only,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()

    def load_dataset(
        self, train_path: str, test_path: str, block_size: int = 64
    ):
        """Load dataset from given datasets paths."""
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=block_size,
        )

        test_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=test_path,
            block_size=block_size,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        return train_dataset, test_dataset, data_collator
