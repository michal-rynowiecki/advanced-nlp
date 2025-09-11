import argparse
from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import myutils


def train(input_texts: List[str], gold_labels: List[int]):
    """
    Predict the final labels on a target dataset. 

    Parameters
    ----------
    input_texts: List[str]
        The data to train on, as a list of strings
    gold_labels: List[int]
        The labels for A trained countvectorizer that will be used to convert the input texts

    Returns
    -------
    word_vectorizer: Countvectorizer
        The countvectorizer that can transform text to be in the right format for the classifier
    classifier: LogisticRegression
        The trained classifier
    """
    word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    train_feats = word_vectorizer.fit_transform(input_texts)

    classifier = LogisticRegression(max_iter=500)
    classifier.fit(train_feats, gold_labels)
    return word_vectorizer, classifier

def predict(data: List[str], vectorizer: CountVectorizer, classifier: LogisticRegression):
    """
    Predict the final labels on a target dataset. 

    Parameters
    ----------
    data: List[str]
        The data to predict on, as a list of strings
    vectorizer: CountVectorizer
        A trained countvectorizer that will be used to convert the input texts
    classifier: LogisticRegression
        The already trained classifier to use for the prediction

    Returns
    -------
    dev_preds: List[int]
        The predictions from the classifier on the data
    """
    dev_feats = vectorizer.transform(data)
    dev_preds = classifier.predict(dev_feats)
    return dev_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to development data.")
    args = parser.parse_args()

    # Read the training data
    train_texts, train_labels = myutils.read_data(args.train_path)
    # Train the classifier
    vectorizer, classifier = train(train_texts, train_labels)
    # Predict on the dev data
    dev_texts, dev_labels = myutils.read_data(args.dev_path)
    dev_predictions = predict(dev_texts, vectorizer, classifier)
    # Print the score (accuracy)
    score = myutils.accuracy(dev_labels, dev_predictions)
    print('Logres ' + args.dev_path + ' ' + '{:.2f}'.format(score))
    
