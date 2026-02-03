import sys 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import random 

from transformer_blocks import Block 

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Non")


import sentencepiece as spm 
# we need to load corpus.txt --> this is our dataset --> we want to train -->sentencepiece tokenizer => How to generate tokens   

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=40, # we want 40 unique words  --> we want 40 diff words
    model_type="bpe"
)

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

ids = sp.encode(text, out_type=int)  # encode --> its a function of SentencePieceTokenizer --> How much ever text or tokens --> it would encode it and assign a unique number.

# ----
# we will get to files:
# 1)tokenizer.model --> this will be trained on our dataset
# 2)tokenizer.vocab     
# we want to use tokenizer.model and  we want to divide our dataset into tokens and assign unique ids 
# ----

# we cannot give this data to LLM -->Bcoz the o/p will come as List 

# If we have to give to LLM as a Input --> we need to keep the data as Tensor Form 
data = torch.tensor(ids, dtype=torch.long) 

# if we want to print and check what kinf of data is there
print(data)

vocab_size = sp.get_piece_size()

# corpus = [
#     "hello friends how are you",
#     "the tea is very hot",
#     "my name is Sagar Hegde",
#     "the roads of Pune are busy",
#     "it is raining in pune",
#     "the train is late again",
#     "i love eating vada pav and drinking tea",
#     "holi is my favorite festival",
#     "diwali vrings lights and sweets",
#     "india won the cricket match"
# ]

# Step1:-
# WRITE A PYTHON CODE -->     "hello friends how are you <END>", for every sentence at the end "<END>".
# WE WANT TO COMBINE THE 10 SENTENCES AND COMBINE IT TOGETHER.

# corpus = [s + " <END> "for s in corpus]
# text = " ".join(corpus) 
# print(text)

# words = list(set(text.split()))
# print(words)

# vocab_size = len(words)
# print(vocab_size)  #44

# word2idx = {w: i for i, w in enumerate(words)}
# # print(word2idx)


# # will get a list of numbers we wont get the words
# data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
# print(data) 
# print(len(data)) #62
# # lets find out the vocabolary of our dataset 
# # vocabolary means --> our llm know how many unique words [how much knowledge LLM knows about words] 

# '''
# data = 
# tensor([19, 38, 39, 41, 37, 36, 14,  4,  6, 16, 10, 36,  8, 15,  6, 33,  1, 36,
#         14, 27, 20, 32, 41, 26, 36, 24,  6, 22, 18,  2, 28,  6, 23,  7, 36, 43,
#         12, 29, 30, 35, 25,  3,  4, 36,  9,  6,  8, 34, 42, 36, 21, 11, 13, 25,
#          5, 36,  0, 40, 14, 31, 17, 36])

# hello
# '''

block_size = 6 # context lenght -->LLM can see last 6 words and then generate the next word
embedding_dim = 32 # for every unique token we have 1D Vector inside that there will be 32 values
n_heads = 2 # we wil use 2 multi head attention layer 
n_layers = 2 # we will use 2 transformer blocks [1 transformer block o/p will become the input of next transformer block ]
lr = 1e-3 # learning rate
epochs = 1500  # 1500 times will train the model

# we will create 1 function for Batches 
# we need to divide our data into batches  --> before giving it to the Model

# def get_batch(batch_size=16):
#     ix = torch.randint(len(data) - block_size, (batch_size,)) # 62-6=56 (0-55), 16
#     # 16 examples (sentences sequences) -->o/p --> [4,12,22,34,51........]
#     # 12 (12,13,14,15,16,17) 1st sequence this will have 6 words i.e 6 tokens 
#     # 4 (4,5,6,7,8,9) 2nd sequence
#     # like this total 16 sequeces  because out batch_size=16
#     x = torch.stack([data][i:i+block_size] for i in ix) # x= i/p
#     y = torch.stack([data][i+1:i+block_size] for i in ix) # y= o/p
#     #x = [token_12, token_13, token_14, token_15, token_16, token_17]
#     #y = [token_13, token_14, token_15, token_16, token_17, token_18]
#     return x, y

def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) # 42,32

        self.position_embedding = nn.Embedding(block_size, embedding_dim) # 6,32
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) # 32,42

     
    def forward(self, idx, targets=None):
        B, T = idx.shape # 16,6

        tok_emb = self.token_embedding(idx) # 16,6,32
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # 16,6,32
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x) # RAW PREDICTIONS 
        loss = None
        if targets is not None:
            B, T, C = logits.shape # 16,6,42
            loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))
        return logits, loss 


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
            return idx 

# Training Model
model=TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 3000 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


# test model
# context = torch.tensor([[word2idx["hello"]]], dtype=torch.long)
# out = model.generate(context, max_new_tokens=15)

# print("\nGenerated text:\n")
# print(" ".join(idx2word[int(i)] for i in out[0]))


# context --> what input sequence we are giving to the model 
# later we want to check the model output 

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

context = torch.tensor([sp.encode("hello")], dtype=torch.long)

out = model.generate(context, max_new_tokens=20)

print("\nGenerated text:\n")

generated_ids = out[0].tolist()
print(sp.decode(generated_ids))