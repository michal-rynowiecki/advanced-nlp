import argparse
import time
import os
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

import myutils
from myutils import Vocabulary

class LSTMModel(torch.nn.Module):
    def __init__(self, embed_dim: int, lstm_dim: int, vocab_dim: int, num_labels: int):
        """
        Model for classification with an LSTM.
        
        This model combines an embedding layer with an LSTM layer, and a
        final classification linear layer.
    
        Parameters
        ----------
        embed_dim: int
            Size of the word embeddings.
        lstm_dim: int
            Hidden size of the LSTM layer.
        vocab_dim: int
            Number of (input) words in the vocabulary.
        num_labels: int
            Vocabulary size of output space (i.e. number of labels).
        """
        super(LSTMModel, self).__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_dim, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.hidden_to_label = torch.nn.Linear(lstm_dim * 2, num_labels)
        self.lstm_dim = lstm_dim
        
    
    def forward(self, inputs: torch.tensor):
        """
        Forward pass, goes through all layers, and return the logits.
    
        Parameters
        ----------
        embed_dim: int
            Size of the word embeddings.
        lstm_dim: int
            Hidden size of the LSTM layer.
        vocab_dim: int
            Number of (input) words in the vocabulary.
        num_labels: int
            Vocabulary size of output space (i.e. number of labels).

        Returns
        logits
            
        -------
        """
        word_vectors = self.word_embeddings(inputs)
        lstm_out, _ = self.lstm(word_vectors)

        backward_out = lstm_out[:,0,-self.lstm_dim:]
        forward_out = lstm_out[:,-1,:self.lstm_dim]
        combined = torch.zeros((len(lstm_out), self.lstm_dim*2), device=lstm_out.device)
        combined[:,:self.lstm_dim] = forward_out
        combined[:,-self.lstm_dim:] = backward_out

        logits = self.hidden_to_label(combined) 
        return logits

def convert_data(texts: List[str], word_vocabulary: Vocabulary, max_len: int, update: bool = False):
    all_data = []
    for line in texts:
        line_indices = []
        for word in line.split(' ')[:max_len]:
            line_indices.append(word_vocabulary.getIdx(word, update))
        
        all_data.append(torch.tensor(line_indices))

    # Convert the data to a torch tensor, where each instance has the same length, 
    # use [UNK] for padding.
    return pad_sequence(all_data, batch_first=True, padding_value=word_vocabulary.getIdx('[UNK]', False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to development data.")

    parser.add_argument("--lstm_dim", type=int, default=50, help="Dimension of the hidden layer of the LSTM.")
    parser.add_argument("--embed_dim", type=int, default=100, help="Dimensions of the input embeddings.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use.")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum length (cutoff) of a sentence.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for.")
    args = parser.parse_args()


    # Read the training data
    train_texts, train_labels = myutils.read_data(args.train_path)
    train_labels = torch.tensor(train_labels) # shape: [10000]
    word_vocabulary = myutils.Vocabulary()
    train_features = convert_data(train_texts, word_vocabulary, args.max_len, update=True) # shape: [10000, 52]
    
    train_feats_batches = myutils.to_batch(train_features, args.batch_size) # shape: [156, 64, 52]
    train_labels_batches = myutils.to_batch(train_labels, args.batch_size)  # shape: [156, 64]
    
    # Read the dev data
    dev_texts, dev_labels = myutils.read_data(args.dev_path)
    dev_features = convert_data(dev_texts, word_vocabulary, args.max_len, update=False)
    dev_feats_batches = myutils.to_batch(dev_features, args.batch_size) # shape: [13, 64, 47]

    # initialize and train the model
    lstm_model = LSTMModel(args.embed_dim, args.lstm_dim, len(word_vocabulary), 2)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=args.lr)
    start_time = time.time()

    # Main training loop
    print("Overview of the model:")
    print(lstm_model)
    print()
    for epoch in range(args.num_epochs):
        print("===========")
        print("Epoch: " + str(epoch))
        lstm_model.train()

        epoch_train_loss = 0.0
        for batch_feats, batch_labels in zip(train_feats_batches, train_labels_batches):
            optimizer.zero_grad()
            logits = lstm_model.forward(batch_feats)
            batch_loss = loss_function(logits, batch_labels)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()

        print("Train loss: {:.4f}".format(epoch_train_loss/len(train_feats_batches)))
        print("Total time: " + str(int(time.time() - start_time)) + ' seconds')

        lstm_model.eval()
        dev_predictions = []
        for dev_feats in dev_feats_batches:
            pred_labels = torch.argmax(lstm_model.forward(dev_feats), 1)
            dev_predictions.extend(pred_labels.tolist())
        score = myutils.accuracy(dev_labels, dev_predictions)
        print('Accuracy on dev data: {:.2f}'.format(score))
        print()
