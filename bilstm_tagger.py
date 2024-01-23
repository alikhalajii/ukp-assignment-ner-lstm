import torch
import numpy as np
from torch import nn

class BiLSTMTagger(nn.Module):
    # Module for sequence tagging using bidirectional LSTM and a pretrained embedding

    def __init__(self, text, label, emb_dim=50, num_layers=1, hidden_size=100, class_weights=None, dropout_rate=0.3):
        super(BiLSTMTagger, self).__init__()
        self.num_labels = len(label.vocab)
        self.class_weights = class_weights

        # apply pretrained vectors to Embedding
        self.embedding = nn.Embedding(len(text.vocab), emb_dim)
        self.embedding.weight = torch.nn.Parameter(text.vocab.vectors, requires_grad=False)

        # Build bidirectional LSTM with linear output layer and cross entropy loss
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.top_layer = nn.Linear(2 * hidden_size, self.num_labels)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        # Class Weight Tensor has not been applied
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        # Save the position of pad word & label
        self.pad_word_id = text.vocab.stoi[text.pad_token]
        self.pad_label_id = label.vocab.stoi[label.pad_token]

    # Define the forward computation
    def forward(self, sentence, labels):
        scores = self.compute_outputs(sentence)
        # Flatten the predicted scores and ground truth labels
        scores = scores.view(-1, self.num_labels)
        labels = labels.view(-1)
        # Class Weight Tensor has not been applied
        return self.loss(scores, labels)

    # Computes the output logits for the given input sentence.
    def compute_outputs(self, sentence):
        embedded = self.embedding(sentence) # Look up the embedding for words in the sentence
        embedded = self.dropout(embedded)   # Apply dropout to the LSTM output
        lstm_out, _ = self.lstm(embedded)   # Apply LSTM to obtain hidden states
        lstm_out = self.dropout(lstm_out)   # Apply dropout to the LSTM output
        out = self.top_layer(lstm_out)      # Apply the top layer to obtain the final output logits

        # Encourage the model to ignore the column of pad tokens during training.
        pad_mask = (sentence == self.pad_word_id).float()
        out[:, :, self.pad_label_id] += pad_mask*10000
        return out

    # Return the best predicted label
    def predict(self, sentence):
        scores = self.compute_outputs(sentence)
        predicted = scores.argmax(dim=2)
        return np.array(predicted.t().cpu())
