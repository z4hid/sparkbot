import nltk

import numpy as np

from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Tokenize a sentence into individual words.

    Args:
        sentence (str): The sentence to be tokenized.

    Returns:
        list: A list of individual words in the sentence.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stems a given word by converting it to its root form using the Porter stemming algorithm.

    Parameters:
        word (str): The word to be stemmed.

    Returns:
        str: The stemmed form of the word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Create a bag of words representation of a tokenized sentence.

    Args:
        tokenized_sentence (list): The tokenized sentence to create the bag of words for.
        words (list): The list of words to compare against the tokenized sentence.

    Returns:
        np.array: A numpy array representing the bag of words.
    """
    
    tokenized_sentence = [token.lower() for token in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    
    for index, w in enumerate(words):
        if w in tokenized_sentence: 
            bag[index] = 1.0

    return bag
