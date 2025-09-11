"""
A basic classifier based on the transformers
(https://github.com/huggingface/transformers) library. It loads a language
model (by default bert-base-cased), and adds a linear layer for prediction.
Needs the path to a training file and a development file.
"""
import argparse
from typing import List, Dict
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

import myutils

class BertModel(torch.nn.Module):
    def __init__(self, lm: str, nlabels: int):
        """
        Model for classification with transformers.

        The architecture of this model is simple, we just have a transformer
        based language model, and add one linear layer to converts it output
        to our prediction.
    
        Parameters
        ----------
        lm : str
            Name of the transformers language model to use, can be found on:
            https://huggingface.co/models
        nlabels : int
            Vocabulary size of output space (i.e. number of labels)
        """
        super().__init__()

        # The transformer model to use
        self.lm = AutoModel.from_pretrained(lm)

        # Find the size of the output of the masked language model
        if hasattr(self.lm.config, 'hidden_size'):
            self.lm_out_size = self.lm.config.hidden_size
        elif hasattr(self.lm.config, 'dim'):
            self.lm_out_size = self.lm.config.dim
        else: # if not found, guess
            self.lm_out_size = 768

        # Create prediction layer
        self.hidden_to_label = torch.nn.Linear(self.lm_out_size, nlabels)

    def forward(self, input: torch.tensor, attention_mask: torch.tensor):
        """
        Forward pass
    
        Parameters
        ----------
        input : torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, max_sent_len).
        mask : torch.tensor
            Mask corresponding to input. shape=(batch_size, max_sent_len)

        Returns
        -------
        output_scores : torch.tensor
            the predictions . shape=(batch_size, number_of_labels)
        """
        # Run transformer model on input
        lm_out = self.lm(input, attention_mask=attention_mask)

        # Keep only the last layer: shape=(batch_size, max_len, DIM_EMBEDDING)
        lm_last_layer = lm_out.last_hidden_state
        # Keep only the output for the first ([CLS]) token: shape=(batch_size, DIM_EMBEDDING)
        cls_output = lm_last_layer[:, 0, :]  
    
        # Conver the cls output to label scores
        logits = self.hidden_to_label(cls_output)

        return logits

def convert_data(texts: List[str], tokenizer: AutoTokenizer, max_len: int):
    """
    This function converts a textual input into token indices. 
    
    Parameters
    ----------
    texts: List[str]
        The input texts to convert.
    tokenizer: AutoTokenizer
        The tokenizer to use
    max_len: int
        Maximum length, which can be used to save memory

    Returns
    -------
    text_feats:
        Matrix of word indices, representing the texts.
    mask:
        Matrix of the same size, filled with 1,0 indicating where the tokens are
    """
    
    all_data = []
    for line in texts:
        all_data.append(tokenizer.encode(line, max_length=max_len, truncation=True, return_tensors='pt')[0])

    # Convert the data to a torch tensor, where each instance has the same length, 
    # use [UNK] for padding.
    text_feats = pad_sequence(all_data, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Create mask
    mask = torch.zeros_like(text_feats)
    for lineIdx, word_idcs in enumerate(all_data):
        mask[lineIdx][:len(word_idcs)] = 1
    return text_feats, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to development data.")

    parser.add_argument("--language_model", default = 'bert-base-cased', help='Language model to use')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use.")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum length (cutoff) of a sentence.")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for.")
    args = parser.parse_args()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.language_model, clean_up_tokenization_spaces=True)

    # Read the training data
    train_texts, train_labels = myutils.read_data(args.train_path)
    train_labels = torch.tensor(train_labels)
    train_features, train_masks = convert_data(train_texts, tokenizer, args.max_len)

    train_feats_batches = myutils.to_batch(train_features, args.batch_size)
    train_mask_batches = myutils.to_batch(train_masks, args.batch_size)
    train_labels_batches = myutils.to_batch(train_labels, args.batch_size)

    # Read the dev data
    dev_texts, dev_labels = myutils.read_data(args.dev_path)
    dev_features, dev_masks = convert_data(dev_texts, tokenizer, args.max_len)

    dev_feats_batches = myutils.to_batch(dev_features, args.batch_size)
    dev_mask_batches = myutils.to_batch(dev_masks, args.batch_size)
    
    # initialize and train the model
    bert_model = BertModel(args.language_model, 2)
    bert_model.to(DEVICE)

    for param in bert_model.parameters():
        param.requires_grad = True
    for param in bert_model.lm.parameters():
        param.requires_grad = True

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr)
    start_time = time.time()

    print("Overview of the model:")
    print(bert_model)
    print()
    for epoch in range(args.num_epochs):
        print("===========")
        print("Epoch: " + str(epoch))
        bert_model.train() 
    
        epoch_train_loss = 0.0
        for batch_feats, batch_masks, batch_labels in zip(train_feats_batches, train_mask_batches, train_labels_batches):
            batch_feats = batch_feats.to(DEVICE)
            batch_masks = batch_masks.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = bert_model.forward(batch_feats, attention_mask=batch_masks)
            batch_loss = loss_function(logits, batch_labels)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()

        print("Train loss: {:.6f}".format(epoch_train_loss/len(train_feats_batches)))
        print("Total time: " + str(int(time.time() - start_time)) + ' seconds')

        bert_model.eval()
        dev_predictions = []
        for dev_feats, dev_masks in zip(dev_feats_batches, dev_mask_batches):
            dev_feats = dev_feats.to(DEVICE)
            dev_masks = dev_masks.to(DEVICE)
            logits = bert_model.forward(dev_feats, attention_mask=dev_masks)
            pred_labels = torch.argmax(logits, 1)
            dev_predictions.extend(pred_labels.tolist())
        score = myutils.accuracy(dev_labels, dev_predictions)
        print('Accuracy on dev data: {:.2f}'.format(score))
        print()

