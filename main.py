import time
import torch
from collections import Counter

# Importing from internal modules
from model import GPTModel
from train import train_model_with_regularization
from data import BPETokenizerSimple, create_corpus, create_dataloader_v1

# -------------------------------
# Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Model Configuration
# -------------------------------
AE_GPT_CONFIG = {
    "vocab_size": 8000,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.3,
    "qkv_bias": True,
    "ln_eps": 1e-5,
    "resid_drop_rate": 0.2,
    "attn_drop_rate": 0.2,
}

# -------------------------------
# Tokenizer Initialization
# -------------------------------
tokenizer = BPETokenizerSimple()
tokenizer.load_vocab_and_merges(
    vocab_path="vocab/vocab.json",
    bpe_merges_path="vocab/bpe_pairs.txt"
)

# -------------------------------
# Corpus Preparation
# -------------------------------
text = create_corpus()

# Train-validation split
train_ratio = 0.80
split_idx = int(train_ratio * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]

# -------------------------------
# DataLoader Preparation
# -------------------------------
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=32,
    max_length=AE_GPT_CONFIG["context_length"],
    stride=AE_GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=32,
    max_length=AE_GPT_CONFIG["context_length"],
    stride=AE_GPT_CONFIG["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# -------------------------------
# Sanity Check: Show Loader Shapes
# -------------------------------
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
    break
print(f"Total train batches: {len(train_loader)}")

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
    break
print(f"Total val batches: {len(val_loader)}")

# -------------------------------
# Token Statistics
# -------------------------------
train_tokens = sum(x.numel() for x, _ in train_loader)
val_tokens = sum(x.numel() for x, _ in val_loader)

print(f"Training tokens: {train_tokens}")
print(f"Validation tokens: {val_tokens}")
print(f"Total tokens: {train_tokens + val_tokens}")

# -------------------------------
# Utility: Most Common Phrases
# -------------------------------
def get_most_common_phrases(text, n=3, top_k=10):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    phrases = [' '.join(ngram) for ngram in ngrams]
    counter = Counter(phrases)
    return counter.most_common(top_k)

print("\nMost Common 3-grams:")
most_common = get_most_common_phrases(text, n=3, top_k=10)
for phrase, count in most_common:
    print(f"{phrase} → {count} times")

# -------------------------------
# Model Initialization & Training
# -------------------------------
start_time = time.time()

model = GPTModel(AE_GPT_CONFIG)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

num_epochs = 20

train_losses, val_losses, tokens_seen = train_model_with_regularization(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=20,
    eval_iter=20,
    start_context="n ṯw ꞽm",  # Seed prompt
    tokenizer=tokenizer
)

end_time = time.time()
exec_time = (end_time - start_time) / 60
print(f"\n✅ Model training completed in {exec_time:.2f} minutes.")

# -------------------------------
# Model Summary & Save
# -------------------------------
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Save model and config
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": AE_GPT_CONFIG,
}, "gpt_model_checkpoint.pt")

print("📦 Model checkpoint saved to 'gpt_model_checkpoint.pt'")
