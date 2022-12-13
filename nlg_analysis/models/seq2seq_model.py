import random
from typing import List, Tuple

import torch
from torch import nn, optim

from nlg_analysis.models.base_model import BaseModel
from nlg_analysis.models.conversation_side import ConversationSide
from nlg_analysis.models.seq2seq_decoder import Seq2SeqDecoder
from nlg_analysis.models.seq2seq_encoder import Seq2SeqEncoder
from nlg_analysis.models.seq2seq_utils import (
    prepareData,
    tensorFromSentence,
    tensorsFromPair,
)


class Seq2SeqModel(BaseModel):
    def __init__(
        self,
        questions_path: str,
        answers_path: str,
        path2encoder: str = None,
        path2decoder: str = None,
    ) -> None:
        super(Seq2SeqModel, self).__init__()
        self.sos_token = 0
        self.eos_token = 1
        self.questions_path = questions_path
        self.answers_path = answers_path
        if path2encoder is not None:
            self.path2encoder = path2encoder
        else:
            self.path2encoder = None
        if path2decoder is not None:
            self.path2decoder = path2decoder
        else:
            self.path2decoder = None

    def generate_transcript(self, questions: List) -> str:
        if self.path2decoder is None and self.path2encoder is None:
            raise ValueError
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_size = 256
        question_side, answer_side, pairs = prepareData(
            self.questions_path, self.answers_path
        )
        encoder = Seq2SeqEncoder(
            question_side.n_words, hidden_size, device=device
        )
        encoder.load_state_dict(torch.load(self.path2encoder))
        decoder = Seq2SeqDecoder(
            hidden_size, answer_side.n_words, dropout_p=0.1, device=device
        )
        decoder.load_state_dict(torch.load(self.path2decoder))

        trans_txt = ""
        for q in questions:
            output_words, attentions = self.evaluate(
                encoder=encoder,
                decoder=decoder,
                sentence=q,
                device=device,
                answer_side=answer_side,
                question_side=question_side,
            )
            trans_txt += q + "\n" + " ".join(output_words[:-1]) + "\n"
        return trans_txt

    @staticmethod
    def approach() -> str:
        return "Seq2Seq"

    def train(
        self,
        encoder_output_path: str,
        decoder_output_path: str,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = 0.01
        hidden_size = 256
        n_iters = 75000
        question_side, answer_side, pairs = prepareData(
            self.questions_path, self.answers_path
        )
        encoder = Seq2SeqEncoder(
            question_side.n_words, hidden_size, device
        ).to(device)
        attn_decoder = Seq2SeqDecoder(
            hidden_size, answer_side.n_words, device, dropout_p=0.1
        ).to(device)

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(
            attn_decoder.parameters(), lr=learning_rate
        )
        training_pairs = [
            tensorsFromPair(
                random.choice(pairs),
                input_side=question_side,
                output_side=answer_side,
                eos_token=self.eos_token,
                device=device,
            )
            for _ in range(n_iters)
        ]
        criterion = nn.NLLLoss()

        for i in range(1, n_iters + 1):
            training_pair = training_pairs[i - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            self.trainIter(
                input_tensor,
                target_tensor,
                encoder,
                attn_decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
                device,
            )

        torch.save(encoder.state_dict(), encoder_output_path)
        torch.save(attn_decoder.state_dict(), decoder_output_path)

    def trainIter(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        max_length=32,
        teacher_forcing_ratio=0.5,
    ) -> float:
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device
        )

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.sos_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = (
            True if random.random() < teacher_forcing_ratio else False
        )

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]

        else:
            # Without teacher forcing:
            # use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                topv, topi = decoder_output.topk(1)
                decoder_input = (
                    topi.squeeze().detach()
                )  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.eos_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def evaluate(
        self,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        sentence: str,
        question_side: ConversationSide,
        answer_side: ConversationSide,
        device: torch.device,
        max_length=32,
    ) -> Tuple[List[str], torch.Tensor]:
        with torch.no_grad():
            input_tensor = tensorFromSentence(
                question_side,
                sentence,
                eos_token=self.eos_token,
                device=device,
            )
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(
                max_length, encoder.hidden_size, device=device
            )

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.sos_token]], device=device)

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.eos_token:
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(answer_side.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[: di + 1]
