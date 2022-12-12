import random
from typing import List

import torch
from torch import nn, optim

from nlg_analysis.models.base_model import BaseModel
from nlg_analysis.models.seq2seq_decoder import Seq2SeqDecoder
from nlg_analysis.models.seq2seq_encoder import Seq2SeqEncoder
from nlg_analysis.models.seq2seq_utils import prepareData, tensorsFromPair


class Seq2SeqModel(BaseModel):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.sos_token = 0
        self.eos_token = 1

    def generate_transcript(self, questions: List) -> str:
        pass

    @staticmethod
    def approach() -> str:
        return "GRU"

    def train(
        self,
        questions_path: str,
        answers_path: str,
        encoder_output_path: str,
        decoder_output_path: str,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = 0.01
        hidden_size = 256
        n_iters = 75000
        input_lang, output_lang, pairs = prepareData(
            questions_path, answers_path
        )
        encoder = Seq2SeqEncoder(input_lang.n_words, hidden_size, device).to(
            device
        )
        attn_decoder = Seq2SeqDecoder(
            hidden_size, output_lang.n_words, device, dropout_p=0.1
        ).to(device)

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(
            attn_decoder.parameters(), lr=learning_rate
        )
        training_pairs = [
            tensorsFromPair(
                random.choice(pairs),
                input_side=input_lang,
                output_side=output_lang,
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
        input_tensor,
        target_tensor,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        criterion,
        device,
        max_length=32,
        teacher_forcing_ratio=0.5,
    ):
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
