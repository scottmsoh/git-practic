import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len,
        vocab_size,
        num_layers,
        d_model,
        d_ff,
        num_heads,
        dropout_ratio,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio

        self.encoder = Encoder(
            self.seq_len,
            self.vocab_size,
            self.num_layers,
            self.d_model,
            self.d_ff,
            self.num_heads,
            self.dropout_ratio,
        )
        self.decoder = Decoder(
            self.seq_len,
            self.vocab_size,
            self.num_layers,
            self.d_model,
            self.d_ff,
            self.num_heads,
            self.dropout_ratio,
        )

        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def _create_padding_mask(self, x):
        padding_mask = torch.where(x==0, 1, 0).unsqueeze(1).unsqueeze(2)    # batch_size, 1, 1, seq_len

        return padding_mask

    def _create_look_ahead_mask(self, x):
        look_ahead_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).unsqueeze(0)   # batch_size, seq_len, seq_len
        padding_mask = self._create_padding_mask(x).squeeze(1)  # batch_size, 1, seq_len

        return torch.max(look_ahead_mask, padding_mask).unsqueeze(1)

    def forward(self, encoder_input, decoder_input):
        encoder_padding_mask = self._create_padding_mask(encoder_input)
        decoder_padding_mask = self._create_padding_mask(decoder_input)
        look_ahead_mask = self._create_look_ahead_mask(x)

        encoder_output = self.encoder(encoder_input, encoder_padding_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, look_ahead_mask, decoder_padding_mask)
        output = self.linear(decoder_output)

        return output
