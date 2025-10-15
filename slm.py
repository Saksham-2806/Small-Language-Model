# slm.py  -- minimal Small Language Model (SLM) from scratch (PyTorch)
# Requirements:
#   pip install torch tiktoken tqdm numpy
# Usage:
#   python slm.py --prepare_data   # tokenizes sample text into train.bin/val.bin
#   python slm.py --train         # train a few steps
#   python slm.py --sample        # generate text

import argparse, os, math, time
import numpy as np
from tqdm import tqdm
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Config / hyperparams
# -----------------------
class Config:
    dataset_text = "tiny_data.txt"   # put your text here (one big text or many concatenated)
    train_bin = "train.bin"
    val_bin = "val.bin"
    vocab_name = "gpt2"              # use tiktoken gpt2 encoding
    dtype = np.uint16
    block_size = 8                 # context length
    batch_size = 4
    n_layer = 4
    n_head = 4
    n_embd = 256
    dropout = 0.1
    lr = 3e-4
    max_iters = 2000
    eval_interval = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

C = Config()

# -----------------------
# Data prep (tokenize & memmap)
# -----------------------
def prepare_data():
    enc = tiktoken.get_encoding(C.vocab_name)
    # read text file (user should provide a dataset or tiny_data.txt will be created)
    if not os.path.exists(C.dataset_text):
        # fallback sample text (tiny)
        s = ("One day a little bird learned to sing. "
             "It sang about the sky and the sun. "
             "Children listened and clapped. "
             "They said the bird was brave and kind. ")
        open(C.dataset_text, "w", encoding="utf-8").write(s*2000)  # make it a bit larger
    else:
        s = open(C.dataset_text, "r", encoding="utf-8").read()

    # encode tokens
    ids = enc.encode_ordinary(s)
    ids = np.array(ids, dtype=C.dtype)
    # split into train/val (90/10)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    np.memmap(C.train_bin, dtype=C.dtype, mode='w+', shape=(len(train_ids,)))[:] = train_ids[:]
    np.memmap(C.val_bin, dtype=C.dtype, mode='w+', shape=(len(val_ids,)))[:] = val_ids[:]
    print(f"Saved train.bin ({len(train_ids)} tokens) and val.bin ({len(val_ids)} tokens).")

# -----------------------
# Batching function (memmap per-batch)
# -----------------------
def get_batch(split):
    dtype = C.dtype
    fname = C.train_bin if split == 'train' else C.val_bin
    data = np.memmap(fname, dtype=dtype, mode='r')
    # pad if too short
    if len(data) < C.block_size + 1:
        data = np.pad(data, (0, (C.block_size + 1) - len(data)), mode='constant')
    ix = torch.randint(len(data) - C.block_size - 1, (C.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+C.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+C.block_size].astype(np.int64)) for i in ix])
    return x.to(C.device), y.to(C.device)

# -----------------------
# Model (tiny GPT-like)
# -----------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(C.block_size, C.block_size)).unsqueeze(0).unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, Cdim = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, n_head, T, head_dim
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        mask = self.mask[:,:,:T,:T].to(att.device)
        att = att.masked_fill(mask==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v   # B, n_head, T, head_dim
        y = y.transpose(1,2).contiguous().view(B, T, Cdim)
        return self.out(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size
        tok = self.tok_emb(idx)            # B, T, C
        x = tok + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)              # B, T, vocab
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------
# Generate function
# -----------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens=100, temperature=1.0):
    enc = tiktoken.get_encoding(C.vocab_name)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -C.block_size:] if idx.size(1) > C.block_size else idx
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-8, temperature)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx

# -----------------------
# Training loop
# -----------------------
def train_loop():
    # prepare model
    enc = tiktoken.get_encoding(C.vocab_name)
    vocab_size = enc.n_vocab
    model = TinyGPT(vocab_size, C.n_layer, C.n_head, C.n_embd, C.block_size, C.dropout).to(C.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr)
    best_val_loss = 1e9

    for it in range(1, C.max_iters+1):
        model.train()
        x, y = get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % C.eval_interval == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch('val')
                _, val_loss = model(xb, yb)
            print(f"iter {it} train_loss {loss.item():.4f} val_loss {val_loss.item():.4f}")

            # sample text
            prompt = "One day"
            idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long).to(C.device)
            sample_ids = generate(model, idx, max_new_tokens=50, temperature=1.0)[0].cpu().numpy().tolist()
            print("SAMPLE:", enc.decode(sample_ids))

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.prepare_data:
        prepare_data()
    elif args.train:
        train_loop()
    elif args.sample:
        # load a small model checkpoint if you saved one; otherwise run a few training steps first
        print("Run training first or load a checkpoint. See script.")
    else:
        parser.print_help()
