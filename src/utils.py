import nltk

import numpy as np

from nltk.stem.porter import PorterStemmer

# Download the 'punkt' package from the NLTK data repository. This is required
# for tokenizing sentences.
nltk.download('punkt')

# Create an instance of the PorterStemmer class. This class provides a method
# for stemming words, which is useful for reducing words to their root form.
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Tokenize a sentence into individual words.

    Args:
        sentence (str): The sentence to be tokenized.

    Returns:
        list: A list of individual words in the sentence.
    """
    # Tokenize the sentence by splitting it into individual words using
    # the word_tokenize function from the NLTK library.
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stems a given word by converting it to its root form using the Porter
    stemming algorithm.
    Parameters:
        word (str): The word to be stemmed.

    Returns:
        str: The stemmed form of the word.
    """
    # Stem the word by converting it to its root form using the Porter
    # stemming algorithm. The stemmer is an instance of the PorterStemmer
    # class, which provides a method for stemming words.
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Create a bag of words representation of a tokenized sentence.
    Args:
        tokenized_sentence (list): The tokenized sentence to create the bag
            of words for.
        words (list): The list of words to compare against the tokenized
            sentence.

    Returns:
        np.array: A numpy array representing the bag of words.
    """
    # Create a numpy array of zeros with the same length as the words list.
    # This array will represent the bag of words.
    bag = np.zeros(len(words), dtype=np.float32)
    
    # Iterate over the words in the words list and check if each word is
    # present in the tokenized sentence. If the word is present, set the
    # corresponding element in the bag array to 1.0.
    for index, w in enumerate(words):
        if w in tokenized_sentence: 
            bag[index] = 1.0
    # Return the bag of words array.
    return bag
