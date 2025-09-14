from typing import List
import json
import torch

def read_data_json(path: str):
    """
    Converts a conll-like (i.e. tab-separated) file into two python list, 
    one for the texts, and one for the labels. It assumes the labels are in 
    the first column, and the texts in the second column. It also assumes
    the labels can be converted to integers.
    
    Parameters
    ----------
    path: str
        The path to read from

    Returns
    -------
    txts: List[str]
        The list of text inputs. Is a list of strings, where each string represents a sentence.
    labels: List[int]
        The list of labels in the same order as the txts.
    """
    txts = []
    labels = []
    for line in open(path):
        tok = json.loads(line)
        # Remove the neutral class
        if tok['label'] == '1':
            continue
        # Adjust the label
        if tok['label'] == '2':
            tok['label'] = '1'
        txts.append(tok['text'])
        labels.append(int(tok['label']))
    return txts, labels

def write_data(txts: list[str], labels: list[int], path: str):
    """
    Writes the provided text list and label list to a conll-like (i.e. tab
    separated) file. It puts the labels first, and text lines afterwards.
    It assumes that the labels can be converted to str format.

    Parameters
    ----------
    txts: list[str]
        list of lines of data
    labels: list[int]
        list of labels for each line, convertable to string
    path: str
        path to save the file

    Returns
    -------
    None
    """
    with open(path, "w") as f:
        for label, line in zip(labels, txts):
            f.write(str(label))
            f.write("\t")
            f.write(line)
            f.write("\n")
    f.close()

def accuracy(predictions: List[str], gold_labels: List[str]):
    """
    Calculate accuracy as a percentage (i.e. * 100).
    
    Parameters
    ----------
    predictions: List[str]
        List of predicted labels
    gold_labels: List[str]
        List of gold labels

    Returns
    -------
    score: float
        The accuracy
    
    """
    cor = sum([pred==gold for pred, gold in zip(predictions, gold_labels)])
    total = len(gold_labels)
    return 100 * cor/total


class Vocabulary():
    def __init__(self):
        """
        Class representing a lookup vocabulary; it converts words to their
        unique index (and vice-versa). 
        """ 
        self.word2idx = {}
        self.idx2word = []
        # add special unk token
        self.getIdx('[UNK]', True)

    def getWord(self, idx: int):
        """
        Convert an index to a word.

        Parameters
        ----------
        idx: int:
            The index of the word to lookup

        Returns
        -------
        word: str
            The surface form of the word.
        """
        return self.idx2word(idx)

    def getIdx(self, word: str, update: bool):
        """
        Convert a word to an index. If the word is not seen before, and update
        is True, it will be added.

        Parameters
        ----------
        word: str
            The surface form of the word to convert.
        update: bool
            Whether to add new words (otherwise it returns the UNK index)

        Returns
        -------
        word: str
            The surface form of the word.
        """
        if update and word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            return len(self.idx2word)-1
        if word not in self.word2idx:
            return 0 # [UNK] is always in 0
        return self.word2idx[word]

    def __len__(self):
        """
        Finds the length of the whole vocabulary

        Returns
        -------
        length: int
            The size of the vocabulary, i.e. the number of words included.
        """
        return len(self.idx2word)

def to_batch(input_tensor: torch.tensor, batch_size: int):
    """
    This is a simplistic implementation of batching, it removes any remainders.
    So if you have a tensor of labels of size 100, and batch size == 32
    it would just remove the last 4 (100-32*3), and put chunks of 32 in the new
    dimension. The resulting tensor would be of size (32,3)
    
    Parameters
    ----------
    input_tensor: torch.tensor
        The input matrix, can be 1 (labels) or 2 dimensional (list of texts, 
        represented by their word indices.
    batch_size: int
        The size of each batch.

    Returns
    -------
    output_tensor: torch.tensor
        The same data as in input_tensor, but transformed to have one more
        dimension of size batch_size, which is the 2nd dimension, so the size is:
        [number_of_batches, batch_size, max_len].
    """
    num_batches = int(len(input_tensor)/batch_size)
    total_size = num_batches * batch_size
    if len(input_tensor.shape) == 2:
        length = input_tensor.shape[1]
        return input_tensor[:total_size].view(num_batches, batch_size, length)
    else:
        return input_tensor[:total_size].view(num_batches, batch_size)

