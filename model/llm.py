import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module, Sequential

from model.transformer import RMSNorm, TransformerBlock


class LanguageModel(Module):
    """Language Model class."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_hidden_dim: int,
        emb_dropout: float = 0.0,
        ffn_bias: bool = False,
    ) -> None:
        super(LanguageModel, self).__init__()

        self.context_size = context_size
        self.token_embedding = Embedding(vocab_size, emb_dim)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer_blocks = Sequential(
            *[TransformerBlock(context_size, emb_dim, num_heads, ffn_hidden_dim, ffn_bias) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(emb_dim)
        self.projection_head = Linear(emb_dim, vocab_size)

        self.token_embedding.weight = self.projection_head.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)
        x = self.emb_dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = self.projection_head(x)

        return x

    @torch.inference_mode()
    def generate(self, inputs: Tensor, max_seq_len: int, temperature: float = 1.0, top_p: int = None) -> Tensor:
        for _ in range(max_seq_len):
            inputs_cond = inputs if inputs.size(1) <= self.context_size else inputs[:, -self.context_size :]

            logits = self(inputs_cond)[:, -1, :]

            probs = F.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs
