import os
import json
import torch
import requests
import pandas as pd
from pandas import json_normalize
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def label_dates(df):
    # Define period boundaries
    periods = [
        {"name": "Predynastic & Early Dynastic", "start": -4300, "end": -2675},
        {"name": "Old Kingdom & First Intermediate", "start": -2675, "end": -1980},
        {"name": "Middle Kingdom & Second Intermediate", "start": -1980, "end": -1539},
        {"name": "New Kingdom & Third Intermediate", "start": -1539, "end": -656},
        {"name": "Late Period & Greco-Roman Egypt", "start": -664, "end": 642}
    ]

    # Initialize the datelabel column with "Unknown"
    df['datelabel'] = "Unknown"

    # Process each row individually
    for i in df.index:
        for period in periods:
            if df.at[i, 'dateNotBefore'] <= period["end"] and df.at[i, 'dateNotAfter'] >= period["start"]:
                df.at[i, 'datelabel'] = period["name"]
                break  # Take the first matching period and stop checking

    return df

# Earlier Egyptian Data
early = pd.read_json("hf://datasets/thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium/train.jsonl", lines=True)

# Late Egyptian Data
late = pd.read_json("hf://datasets/thesaurus-linguae-aegyptiae/tla-late_egyptian-v19-premium/train.jsonl", lines=True)

# Demotic Egyptian Data
# URL for the Hugging Face Datasets API
url = "https://huggingface.co/datasets/thesaurus-linguae-aegyptiae/tla-demotic-v18-premium/raw/main/train.jsonl"

# Make the request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Split the text by lines because it's a JSONL file
    lines = response.text.strip().split('\n')

    # Parse each line as JSON
    data = [json.loads(line) for line in lines]

    # Now each item is a normal JSON object you can work with
    dfs = [json_normalize(item) for item in data]

    # If you want a single DataFrame
    demotic = pd.concat(dfs, ignore_index=True)

demotic['dateNotBefore'] = demotic['dateNotBefore'].replace('', 0)
demotic['dateNotAfter'] = demotic['dateNotAfter'].replace('', 0)
demotic['dateNotBefore'] = demotic['dateNotBefore'].astype(int)
demotic['dateNotAfter'] = demotic['dateNotAfter'].astype(int)

# Apply label_dates to all data
early = label_dates(early)
late = label_dates(late)
demotic = label_dates(demotic)

# Create Corpus
def create_corpus():
    datasets = [early, late, demotic]
    text = ""

    for dataset in datasets:
        for row in dataset['transliteration']:
            text += " " + row  # just add the sentence

    # Uncomment to create the text file containning the corpus
    #with open("text.txt", "w", encoding="utf-8") as file:
        #file.write(text)
        
    return text


tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["<|endoftext|>"])

tokenizer.train(files=["data/text.txt"], trainer=trainer)

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
