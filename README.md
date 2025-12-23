# Prompt-Based-Trauma

Property-Based Trauma (PBT) focuses on verifying program correctness by testing general properties across a wide range of automatically generated inputs instead of fixed test cases.

---

## Overview 

This repository contains scripts and utilities for prompt-based trauma analysis and localization using property-based testing (PBT) ideas combined with sequence-to-sequence Transformer models. The system is designed as a two-stage pipeline:

- **Speech-to-text transcription** using a fine-tuned Whisper model
- **Text-based trauma localization and classification** using a T5-based model

The project includes training, inference, sentence generation, and evaluation utilities to systematically test and measure model behavior across automatically generated and real-world trauma-related prompts.

## Model Pipeline and Methodology 

1. **Whisper-based Transcription**

   - A Whisper model is fine-tuned to transcribe spoken casualty or first-person injury descriptions.
   - The fine-tuned Whisper model focuses on robustness to noisy environments, informal speech, and incomplete or distressed utterances.
   - The output of Whisper is a clean textual transcription that serves as the input prompt for the localization model.

2. **T5-based Trauma Localization

   - A T5-base model is used for trauma localization and structured prediction.
   - Training consists of two stages:
     - **Pretraining** using span corruption (T5 objective) to improve domain adaptation and language robustness.
     - **Fine-tuning** on a curated trauma dataset where each input sentence maps to structured region-wise labels (e.g., Head, Torso, Upper Extremities, Lower Extremities).
   - The model learns to parse free-form natural language descriptions, identify relevant injury cues, and produce consistent, logically aligned trauma localization outputs.
   - On our evaluation dataset, the fine-tuned T5 achieves ~98% accuracy measured using region-wise and sentence-level correctness metrics.

## Features ✨

- Fine-tuning and training utilities for T5-based models
- Pretraining support using span corruption objectives
- Whisper-to-T5 inference pipeline
- Prompt-based inference and prediction generation
- Accuracy and evaluation tools for structured outputs
- Property-based testing utilities for automatically generated sentences
- Helper scripts for sentence generation, augmentation, and sorting

## Quickstart ⚡

1. Clone the repository and set up a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Run T5 training or fine-tuning:

```bash
python -m pbt_localization.train_t5_new
```

3. Run inference using a trained checkpoint:

```bash
python -m pbt_localization.inference --model /path/to/checkpoint
```

4. Evaluate predictions:

```bash
python -m pbt_localization.accuracy --preds preds.json --labels labels.json
```

Most scripts support a `--help` flag to inspect configurable arguments such as batch size, learning rate, dataset paths, and checkpoint directories.

## Project Structure 

```
pbt_localization/
├── train_t5_new.py     # Training and fine-tuning utilities for T5
├── t5_pretrain2.py     # Span-corruption pretraining and auxiliary scripts
├── inference.py        # Inference using trained checkpoints
├── accuracy.py         # Evaluation and accuracy metrics
├── sentence_gen.py     # Prompt and sentence generation utilities
├── sortsent.py         # Sentence sorting and helper tools
```

## Reproducing Experiments 

To reproduce results:

- Prepare datasets in the expected input-output format
- Configure training hyperparameters via CLI arguments or config files
- Run span-corruption pretraining if starting from a base T5 checkpoint
- Fine-tune on the trauma localization dataset
- Use the generated checkpoints for inference and evaluation
- Ensure consistency between training and evaluation label formats to maintain reported accuracy


## License & Contact 

Add a `LICENSE` file to specify usage and distribution terms. For questions, issues, or discussions, please open an issue in the repository.

---
