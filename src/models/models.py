import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import sys
sys.path.append('../data')
from sentiment_data import *


class SentimentClassifier(object):

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralNet(nn.Module):

    def __init__(self, embeddings: nn.Embedding, hidden_dim: int):
        super().__init__()
        
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        in_dim = embeddings.weight.size()[1]
        self.linear1 = nn.Linear(in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.clf = nn.Linear(self.hidden_dim, 2)
        
    def forward(self, in_indices) -> torch.Tensor:
        """Takes a list of words and returns a tensor of class predictions.

        Steps:
          1. Embed each word using self.embeddings.
          2. Take the average embedding of each word.
          3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
          4. Return a tensor representing a log-probability distribution over classes.
        """

        embed = self.embeddings(in_indices)

        meaned = embed.mean(axis=0)
        
        linear1 = self.linear1(meaned)
        
        relued = self.relu(linear1)
        
        output = self.clf(relued)
        
        return output


class NeuralSentimentClassifier(SentimentClassifier):
    """
    wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, trained, embeddings: int):
        
        self.embeddings = embeddings
        
        self.trained_model = trained

    def predict(self, ex_words: List[str]) -> int:
        
        def find_index(word):
            
            a = self.embeddings.word_indexer.index_of(word)

            if a == -1:
                return 1
            else: 
                return a
        
        input_indices = torch.tensor([find_index(word)  for word in ex_words])
        
        out = self.trained_model(input_indices)
        
        return torch.argmax(out,).item()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args 
    :param train_exs: training examples
    :param dev_exs: development set for evaluation
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
   
    embeddings = word_embeddings.get_initialized_embedding_layer()
    lr = args.lr
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_size
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device=torch.device('cpu')
    
    def find_index(word):
        
        a = word_embeddings.word_indexer.index_of(word)
        
        if a == -1:
            return 1
        else: 
            return a
    
    model = NeuralNet(embeddings, hidden_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    random.shuffle(train_exs)

    for epoch in range(num_epochs):
        model.train()
        
        for ex in train_exs:
            
            input_indices = torch.tensor([find_index(word)  for word in ex.words])
            
            input_label = torch.tensor(ex.label)
     
            preds = model(input_indices)
        
            loss = criterion(preds, input_label)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            

    return NeuralSentimentClassifier(model, word_embeddings)