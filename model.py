from dataclasses import dataclass
import torch
import torch.nn as nn
from fast_transformers.builders import TransformerDecoderBuilder
from fast_transformers.masking import TriangularCausalMask
import miditok
import constants
from miditok import REMI, TokenizerConfig
from tokenization_utils import get_dataloader
from torch.nn import TransformerDecoderLayer, TransformerDecoder, Transformer


@dataclass
class TransformerConfig:
    embeded_dim = 512
    d_model=512
    nhead=8
    num_layers=6
    dim_feedforward=2048
    dropout=0.1
    activation=nn.GELU()
    layer_norm_eps=1e-05
    batch_first=True
    norm_first=False
    device="cuda"
    vocab_size=30000
    seq_len=constants.BLOCK_SIZE
    dtype=torch.bfloat16
    


def build_decoder(config):

    # Initialize the model
    decoder_layer = TransformerDecoderLayer(
        d_model=config.d_model, 
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation=config.activation,
        layer_norm_eps=config.layer_norm_eps,
        batch_first=config.batch_first,
        norm_first=config.norm_first,
        device=config.device, 
        dtype=config.dtype)
    
    layer_norm = nn.LayerNorm(config.embeded_dim, dtype=config.dtype)
    decoder = TransformerDecoder(decoder_layer, num_layers=config.num_layers, norm=layer_norm)
    return decoder
    

class MusicTransformer2(nn.Module):
    def __init__(self, config):
        super(MusicTransformer2, self).__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embeded_dim, dtype=config.dtype)
        self.decoder = build_decoder(config)
        self.output_layer = nn.Linear(config.embeded_dim, config.vocab_size, dtype=config.dtype)
        self.tgt_mask = Transformer.generate_square_subsequent_mask(config.seq_len).to(config.device)

    
    def forward(self, x, tgt_key_padding_mask):
        config = self.config
        
        # Get the embeddings
        x_embed = self.embedding(x)

        # Zero the memory
        memory = torch.zeros_like(x_embed).to(config.device)
        
        # Apply the Transformer decoder
        decoder_output = self.decoder(x_embed, memory, tgt_mask=self.tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # Pass through the output layer
        output = self.output_layer(decoder_output)

        return output


"""
    def generate(self, x, tgt_key_padding_mask, max_tokens):
        config = self.config

        memory = self.forward(x, tgt_key_padding_mask)

        for _ in range(max_tokens):
            x_embed = memory[-1, :, :]
            memory = memory[:-1, :, :]
            logits = self.decoder(x_embed, memory)
            memory = memory.vstack(logits)


        return torch.argmax(memory)
"""

"""
@dataclass
class TransformerConfig:
    num_layers: int = 6  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    embedding_dim: int = 512  # Dimension of the embedding
    feedforward_dim: int = 2048  # Dimension of the feedforward network
    dropout_rate: float = 0.1  # Dropout rate
    activation: str = "gelu"  # Activation function
    self_attention_type: str = "reformer"  # Type of self-attention

    def to_kwargs(self):
        return {
            'n_layers': self.num_layers,
            'n_heads': self.num_heads,
            'query_dimensions': self.embedding_dim // self.num_heads,
            'value_dimensions': self.embedding_dim // self.num_heads,
            'feed_forward_dimensions': self.feedforward_dim,
            'dropout': self.dropout_rate,
            'activation': self.activation,
            'self_attention_type': self.self_attention_type
        }



# Define the Transformer Decoder model
class MusicTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.decoder = TransformerDecoderBuilder.from_kwargs(**config.to_kwargs()).get()
        self.output_layer = nn.Linear(config.embedding_dim, vocab_size)
        
    def forward(self, x, memory=None):
        
        # Get the embeddings
        x_embed = self.embedding(x)

        # Not using memory
        if memory is None:
            memory = torch.zeros_like(x_embed)

        # Apply the Transformer decoder
        decoder_output = self.decoder(x_embed, memory=memory)
        
        # Pass through the output layer
        output = self.output_layer(decoder_output)

        return output
"""