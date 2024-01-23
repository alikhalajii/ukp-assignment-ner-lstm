# Named Entity Recognition using Bidirectional LSTM

## Introduction

This repository contains the implementation of a Named Entity Recognition Tagger using a bidirectional Long Short-Term Memory (LSTM) network. The model is designed to identify named entities in a sequence of text, such as names of people, organizations, locations, etc.



## Implementation Limitation

1. **Word Embeddings** A pre-trained word embeddings.
2. **Architecture** A idirectional LSTM with one single 100-dimensional hidden layer.
3. **Training Data Set** 
4. **Batch Size:** The batch size is set to 1. 
5. **Crossentropy-loss and the Adam optimizer**
6. **External Libraries** Native Python libraries, PyTorch, numpy, pandas and matplotlib.



## Requirements (important)

**Place the pre-trained word embedding file 'glove.6B.50d.txt' into the "data" folder.**

**Before running the code, ensure that you have installed the required dependencies in the specified versions.**

! pip install -r requirements.txt


## Run

! python3 main.py



- Additionally, under the "result" folder, you will find the log from my most recent program run.



## TL;DR: Possible Improvements beyond the Task Assignment

1. **Hyperparameter Tuning:** Experimenting with different hyperparameter configurations, including batch size, number of epochs, and hidden layer dimensions, may impact model performance.

2. **Class Weights:** Applying class weights can help address class imbalances in the dataset.

3. **CRF Layer Addition:** Conditional Random Fields (CRF) can be added as an additional layer on top of the LSTM to model dependencies between consecutive labels in a sequence.

4. **Embedding Improvements:** Experiment with more advanced word embeddings (like BERT or ELMo) or contextual embeddings for a better representation of words.

5. **Data Augmentation:** Increase the diversity of the training data through data augmentation techniques.

*And More...*

