# AegyptusTranslit1
[![FreePalestine.Dev](https://freepalestine.dev/header/1)](https://freepalestine.dev)
A GPT-2 based language model trained from scratch on transliterations of Ancient Egyptian texts, using custom tokenization optimized for linguistic features found in hieroglyphic transliteration.

## Overview

- **Architecture**: GPT-2 (custom configuration)
- **Tokenizer**: Byte-level BPE (custom-trained)
- **Language**: Ancient Egyptian (transliterated in Latin script)
- **Vocabulary size**: 6,475
- **Training corpus**: ~30,000 lines of fully intact, unambiguously readable transliterated sentences
- **Training steps**: ~500 steps over 20 epochs

## Intended Use

- Research in ancient Egyptian linguistics
- Automatic completion or generation of transliterated hieroglyphic texts

## Model

The model is available on Hugging Face: [AegyptusTranslit1](https://huggingface.co/YOUR_USERNAME/AegyptusTranslit1)

## Training Data

Data sourced from the [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de/home):

- [Earlier Egyptian original v18 premium](https://huggingface.co/datasets/thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium)
- [Demotic v18 premium](https://huggingface.co/datasets/thesaurus-linguae-aegyptiae/tla-demotic-v18-premium)
- [Late Egyptian v19 premium](https://huggingface.co/datasets/thesaurus-linguae-aegyptiae/tla-late_egyptian-v19-premium)

> Thesaurus Linguae Aegyptiae, Original Earlier Egyptian sentences, corpus v18, premium, v1.1, 2/16/2024 ed. by Tonio Sebastian Richter & Daniel A. Werning on behalf of the Berlin-Brandenburgische Akademie der Wissenschaften and Hans-Werner Fischer-Elfert & Peter Dils on behalf of the Sächsische Akademie der Wissenschaften zu Leipzig.

## Training Progress

| Epoch | Step | Train Loss | Val Loss |
|-------|------|------------|----------|
| 1 | 0 | 7.884 | 7.942 |
| 2 | 40 | 3.949 | 4.032 |
| 3 | 70 | 3.663 | 3.775 |
| 4 | 90 | 3.551 | 3.671 |
| 5 | 120 | 3.462 | 3.587 |
| 6 | 140 | 3.407 | 3.561 |
| 7 | 170 | 3.346 | 3.524 |
| 8 | 190 | 3.331 | 3.518 |
| 9 | 220 | 3.284 | 3.500 |
| 10 | 240 | 3.264 | 3.496 |
| 11 | 270 | 3.208 | 3.483 |
| 12 | 290 | 3.177 | 3.465 |
| 13 | 320 | 3.127 | 3.460 |
| 14 | 340 | 3.095 | 3.452 |
| 15 | 370 | 3.058 | 3.447 |
| 16 | 390 | 3.020 | 3.443 |
| 17 | 420 | 2.981 | 3.422 |
| 18 | 440 | 2.952 | 3.413 |
| 19 | 470 | 2.885 | 3.398 |
| 20 | 490 | 2.845 | 3.391 |

## Acknowledgments

- [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de/home) — for providing the high-quality transliterated Ancient Egyptian text corpora
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka — used as a reference for implementing the GPT-2 architecture and training pipeline

## License

Apache-2.0
