# transformer_pytorch

This is a PyTorch implementation of the Transformer model using the [T5 transformer model](https://arxiv.org/abs/1910.10683) for Neural Machine Translation. For a detailed explanation of the model, please refer to the [blog post](https://towardsdatascience.com/transformers-141e32e69591) or check the arXiv paper. The code is based on the [PyTorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) and the [Hugging Face tutorial](https://huggingface.co/course/chapter7/4?fw=pt). The code is written in a modular way, so that it can be easily extended to other tasks.

## Simple Exemplary Usage
First install the requirements:
```
conda env create --file env.yml -n transformer_pytorch
conda activate transformer_pytorch
```

Then run the training script:
```
python transformer_tutorial.py
```

## Exemplary Execution of main.py
```
python main.py --epochs 30 --train --debug --n_samples 1000000
```

## Extension
You can play around with the flags to include more datapoints or to train on a different language pair. The code is written in a modular way, so that it can be easily extended to other tasks.