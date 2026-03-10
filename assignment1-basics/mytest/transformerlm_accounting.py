vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

"""
transformerLM

token_embedding (vocab_size, d_model)

transformer_block * num_layers
    ln1 (d_model)
    mha 
        q,k,v,o (d_model, d_model) * 4
    ln2 (d_model)
    ffn(swiglu)
        w1w2w3 (d_model, d_ff)

ln_final (d_model)

lm_head (d_model, vocab_size)
"""

token_embeddings = vocab_size * d_model
ln1 = d_model
ln2 = d_model
ln_final = d_model
lm_head = d_model * vocab_size
mha = d_model * d_model * 4
ffn = d_model * d_ff * 3
transformer_block = ln1 + mha + ln2 + ffn

total_param = token_embeddings + transformer_block * num_layers + ln_final + lm_head
total_param_float32 = total_param * 4 / 1024 / 1024 / 1024
print("------------------------------------ans (a) params mem ------------------------------------")
print(total_param)
print(total_param_float32)

print("------------------------------------ans (b) params flops ------------------------------------")
token_embeddings = 0
ln1 = d_model * context_length
ln2 = d_model * context_length
ln_final = d_model * context_length
lm_head = 0
mha = context_length * d_model * d_model * 2 * 3 + context_length * context_length * d_model * 2 + context_length  * d_model * context_length * 2
ffn = context_length * d_ff * d_model * 2 * 3
transformer_block = ln1 + mha + ln2 + ffn
total_flops = token_embeddings + transformer_block * num_layers + ln_final + lm_head
model_list = [(transformer_block, "transformer_block"), (mha, "mha"), (ffn, "ffn")]
model_list.sort(key=lambda x : x[0])
print(model_list)

print(total_flops)

"""
attention  O(n^2d)
FFN  O(nd^2)
"""

