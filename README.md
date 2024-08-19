# LLM-python-gpt

## Overview

This project was made by following along Elliot Arledge and his course on large language models. 
link to video: https://youtu.be/UU1WVnMk4E8?si=zLO4kaycug2ZuFu2

This repository contains a PyTorch implementation of a GPT (Generative Pre-trained Transformer) language model. The model is built from scratch and includes functionalities for both training and generating text. The repository includes two main Python scripts:

1. **Chatbot.py**: Defines the architecture of the GPT model and includes methods for forward propagation, loss calculation, and text generation.
2. **Training.py**: Responsible for loading the model, training it on a given dataset, and saving the trained model.

## Files

### 1. `chatbot.py`
This script contains the definition of the GPT language model. The key components include:

- **Head**: Implements one head of self-attention.
- **MultiHeadAttention**: Combines multiple heads of self-attention in parallel.
- **FeedForward**: A simple feedforward neural network layer.
- **Block**: Represents a single Transformer block consisting of multi-head self-attention and feedforward layers.
- **GPTLanguageModel**: The main model class which stacks multiple Transformer blocks together. It includes methods for forward pass and text generation.

**Dependencies**:
- `torch`
- `torch.nn`
- `torch.nn.functional`

### 2. `training.py`
This script is used for training the GPT model. It includes:

- Argument parsing to set hyperparameters like batch size.
- Functions for loading and preprocessing data.
- A training loop that performs forward propagation, calculates the loss, backpropagates the error, and updates the model's weights.
- Saving the trained model using `pickle`.

**Training Process**:
- The script reads text data, encodes it, and splits it into training and validation sets.
- The model is trained over a number of iterations, with periodic evaluation on the validation set.
- After training, the model is saved to a file for future use.

**Dependencies**:
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `pickle`
- `argparse`
- `mmap`

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch, CUDA
- A CUDA-enabled GPU is recommended for faster training.




## Usage/Examples
### Training the model
To train the model, run the training.py with the desired batch size:
```bash
python training.py -batch_size <batch_size>
```

### Generating Text
After training, you can use the chatbot.py to generate text:
```bash
python model_script.py
```
You will be prompted to input a starting text (prompt), and the model will generate a continuation based on the training data.

