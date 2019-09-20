# Introduction
This is a project I did for GSI Technology internship 2019. 
This repo provides an implementation of the Gemini network for binary code similarity detection in [this paper](https://arxiv.org/abs/1708.06525).

# Getting Started 

## Preparation and Data
Unzip the data by running:
```bash
unzip data.zip
```

The network is written using Tensorflow 1.4 in Python 2.7. You can install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Model Implementation
The model is implemented in `graphnnSiamese.py`.

Run the following code to train the model:
```bash
python train.py
```
or run `python train.py -h` to check the optional arguments.

After training, run the following code to evaluate the model:
```bash
python eval.py
```
or run `python eval.py -h` to check the optional arguments.

# Visualization
The graphEmbeddings notebook contains details about attemp to visualize embeddings in Tensorflow Projector (t-SNE.) In the notebook it uses the model that I trained (included in the repo.)

# Blogs 
* [Application of AI to Cybersecurity - Part 1](https://medium.com/gsi-technology/application-of-ai-to-cybersecurity-part-1-68d252fafdd5)
* [Application of AI to Cybersecurity - Part 2](https://medium.com/gsi-technology/application-of-ai-to-cybersecurity-part-2-3e27ae468fa5)
* [Application of AI to Cybersecurity - Part 3](https://medium.com/gsi-technology/application-of-ai-to-cybersecurity-part-3-19659bdb3422)

# Reference/Credits
* [dnn-binary-code-similarity](https://github.com/xiaojunxu/dnn-binary-code-similarity)
