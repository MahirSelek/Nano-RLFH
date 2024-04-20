import torch 
from .decoder import TransformerBlock

if __name__=="__main__":

    seq_len = 10  
    dim_emb = 64  
    attn_num_heads = 8  
    ffn_hidden_dim = 128  
    transformer_block = TransformerBlock(seq_len, dim_emb, attn_num_heads, ffn_hidden_dim)

    batch_size = 4
    input_tensor = torch.randn(batch_size, seq_len, dim_emb)

    output_tensor = transformer_block(input_tensor)

    print("Output Tensor Shape:", output_tensor.shape)
