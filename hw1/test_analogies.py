"""
Code for Problems 2 and 3 of HW 1.
"""
from typing import Dict, List, Tuple

import numpy as np

from sys import exit #for debug

from numpy.linalg import norm

from embeddings import Embeddings


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Problem 3b: Implement this function.

    Computes the cosine similarity between two matrices of row vectors.

    :param x: A 2D array of shape (m, embedding_size)
    :param y: A 2D array of shape (n, embedding_size)
    :return: An array of shape (m, n), where the entry in row i and
        column j is the cosine similarity between x[i] and y[j]
    """

    x_norm = norm(x, axis=1, keepdims=True)  # Shape: (m, 1)
    y_norm = norm(y, axis=1, keepdims=True)  # Shape: (n, 1)
    
    return (np.dot(x, y.T)) / (x_norm * y_norm.T)

    #return (np.dot(x , y.T)) / (norm(x) * norm(y))
    #raise NotImplementedError("Problem 3b has not been completed yet!")


def get_closest_words(embeddings: Embeddings, vectors: np.ndarray,
                      k: int = 1) -> List[List[str]]:
    """
    Problem 3c: Implement this function.

    Finds the top k words whose embeddings are closest to a given vector
    in terms of cosine similarity.

    :param embeddings: A set of word embeddings
    :param vectors: A 2D array of shape (m, embedding_size)
    :param k: The number of closest words to find for each vector
    :return: A list of m lists of words, where the ith list contains the
        k words that are closest to vectors[i] in the embedding space,
        not necessarily in order
    """

    # print(len(embeddings.indices)) 
    # print(embeddings.vectors.shape)  

    simmat=cosine_sim(vectors, embeddings.vectors)
    closestk_indexes = np.argsort(simmat, axis=1, kind="heapsort")[:, -k:]
    closestk_words = [[list(embeddings.indices.keys())[i] for i in row] for row in closestk_indexes]
    
    return closestk_words
    #raise NotImplementedError("Problem 3c has not been completed yet!")


# This type alias represents the format that the testing data should be
# deserialized into. An analogy is a tuple of 4 strings, and an
# AnalogiesDataset is a dict that maps a relation type to the list of
# analogies under that relation type.

AnalogiesDataset = Dict[str, List[Tuple[str, str, str, str]]]


def load_analogies(filename: str) -> AnalogiesDataset:
    """
    Problem 2b: Implement this function.

    Loads testing data for 3CosAdd from a .txt file and deserializes it
    into the AnalogiesData format.

    :param filename: The name of the file containing the testing data
    :return: An AnalogiesDataset containing the data in the file. The
        format of the data is described in the problem set and in the
        docstring for the AnaslogiesDataset type alias
    """
    analogies_dict = {}
    relation = None
    for line in open(filename, 'r'):
        line = line.lower() #added lowercase because the stupid country capitals aren't in sentnce case in the glove.txt
        if line[0] == ':':
            relation = line.strip(": \n")
            analogies_dict[relation] = []
        else:
            analogy = tuple(line.strip().split()) 
            analogies_dict[relation].append(analogy)
    
    return analogies_dict
    #raise NotImplementedError("Problem 2b has not been completed yet!")


def run_analogy_test(embeddings: Embeddings, test_data: AnalogiesDataset,
                     k: int = 1) -> Dict[str, float]:
    """
    Problem 3d: Implement this function.

    Runs the 3CosAdd test for a set of embeddings and analogy questions.

    :param embeddings: The embeddings to be evaluated using 3CosAdd
    :param test_data: The set of analogies with which to compute analogy
        question accuracy
    :param k: The "lenience" with which accuracy will be computed. An
        analogy question will be considered to have been answered
        correctly as long as the target word is within the top k closest
        words to the target vector, even if it is not the closest word
    :return: The results of the 3CosAdd test, in the format of a dict
        that maps each relation type to the analogy question accuracy
        attained by embeddings on analogies from that relation type
    """
    acc_dict = {}

    for relation, analogylists in test_data.items():
        num_correct = 0

        for A, B, C, target in analogylists:

            # print(A)
            # print(B)
            # print(C)
            # print(target)
            D_vector = embeddings[[B]] - embeddings[[A]] + embeddings[[C]]
            D_kclosest = get_closest_words(embeddings, D_vector, k)
            # print(D_kclosest[0])
            # exit()

            if target in D_kclosest[0]:
                num_correct+=1
        
        relation_acc = num_correct/len(analogylists)
        acc_dict[relation] = relation_acc

    return acc_dict
    #raise NotImplementedError("Problem 3d has not been completed yet!")
