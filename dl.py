import torch
import torch.nn as nn


def scaled_dot_product_attention(query, key, value, mask):
    # query: batch, num_heads, seq_len, head_dim
    qk = query @ key.permute(0, 1, 3, 2)   # torch.bmm(query, query.permute(0, 1, 3, 2))
    scaled_qk = qk / torch.sqrt(query.shape[-1])

    if mask is not None:
        scaled_qk += (mask * -1e9)

    attention_distribution = F.softmax(scaled_qk, dim=-1)
    attention_value = attention_distribution@value

    return attention_value


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = self._encoding(self.seq_len, self.d_model)
    
    def _embedding(self, position_index: int, i: int, d_model: int):
        angle = 1 / (10000 ** (2*(i//2) / d_model))
        
        return position_index * angle

    def _position_encoding(self, seq_len, d_model):
        _embedding = self._embedding(
            position_index=torch.arange(seq_len, dtype=torch.float32).unsqueeze(1),
            i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
            d_model=d_model,
        )

        sines = torch.sin(_embedding[:, ::2])
        cosines = torch.cos(_embedding[:, 1::2])

        embedding = torch.zeros(_embedding.shape)
        embedding[:, 0::2] = sines
        embedding[:, 1::2] = cosines

        return embedding
    
    def _encoding(self, seq_len, d_model):
        return self._position_encoding(seq_len, d_model).unsqueeze(0)

    def forward(self, x):
        return x + self.positional_encoding[:x.size(1)]


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model%num_heads == 0

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        query = self.query(query)           # batch, seq_len, dim
        key = self.key(key)
        value = self.value(value)

        batch_size, seq_len = query.shape[:2]
        query = query.reshape(batch_size, seq_len, self.num_heads, -1)  # batch, seq_len, num_heads, head_dim
        key = key.reshape(batch_size, seq_len, self.num_heads, -1)
        value = value.reshape(batch_size, seq_len, self.num_heads, -1)

        query = query.permute(0, 2, 1, 3)   # batch, num_heads, seq_len, head_dim
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_value = scaled_dot_product_attention(query, key, value, mask)   # batch, num_heads, seq_len, head_dim
        attention_value = attention_value.permute(0, 2, 1, 3)                     # batch, seq_len, num_heads, head_dim
        attention_value = attention_value.reshape(batch_size, seq_len, -1)        # batch, seq_len, dim

        output = self.out(attention_value)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_ratio):
        super().__init__()

        self.multi_head_attention = MultiheadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask):
        x_multi_head_output = self.multi_head_attention(x, x, x, padding_mask)
        x_multi_head_output = self.dropout1(x_multi_head_output)
        x = self.layer_norm1(x + x_multi_head_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        output = self.layer_norm2(x + ffn_output)

        return output


class Encoder(nn.Module):
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

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.seq_len, self.d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                self.d_model,
                self.d_ff,
                self.num_heads,
                self.dropout_ratio,
            ) for _ in range(self.num_layers)
        ])

    def forward(self, x, padding_mask=None):       # (batch, seq_len)
        x = self.embedding(x)   # (batch, seq_len, dim)
        x *= (self.d_model ** 0.5)
        x += self.positional_encoding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask=None)
        output = x

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_ratio):
        super().__init__()

        self.multi_head_attention1 = MultiheadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.multi_head_attention2 = MultiheadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        x_multi_head_output = self.multi_head_attention1(x, x, x, mask=look_ahead_mask)
        x_multi_head_output = self.dropout1(x_multi_head_output)
        x = self.layer_norm1(x + x_multi_head_output)

        x_multi_head_output = self.multi_head_attention2(x, encoder_output, encoder_output, mask=padding_mask)
        x_multi_head_output = self.dropout2(x_multi_head_output)
        x = self.layer_norm2(x + x_multi_head_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output)
        output = self.layer_norm3(x + ffn_output)

        return output


class Decoder(nn.Module):
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

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.seq_len, self.d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                self.d_model,
                self.d_ff,
                self.num_heads,
                self.dropout_ratio,
            ) for _ in range(self.num_layers)
        ])
    
    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x)
        x *= (self.d_model ** 0.5)
        x += self.positional_encoding(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, look_ahead_mask, padding_mask)
        output = x

        return output


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

    def forward(self, x, encoder_input, decoder_input):
        encoder_padding_mask = self._create_padding_mask(encoder_input)
        decoder_padding_mask = self._create_padding_mask(decoder_input)
        look_ahead_mask = self._create_look_ahead_mask(x)

        encoder_output = self.encoder(encoder_input, encoder_padding_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, look_ahead_mask, decoder_padding_mask)
        output = self.linear(decoder_output)

        return output
