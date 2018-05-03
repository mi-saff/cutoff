from collections import defaultdict
from ast import literal_eval as make_tuple
import re
import os.path

FEATURE_SIZE = 90

def get_file_length(filename):
    with open(filename, "r") as doc:
        return sum(1 for line in doc)

def normalize_length_dict(length_dict):
    new_length_dict = defaultdict(list)
    max_size = max(max(length_dict[key]) for key in length_dict)
    for key in length_dict:
        new_length_dict[key] = [float(x) / float(max_size) for x in length_dict[key]]
    return new_length_dict


def query_length(queryname):
    with open("./query_list.tsv") as fp:
        lines = fp.readlines()
        for line in lines:
            token_list = line.split(" ")
            if token_list[0] == queryname:
                return token_list[1]

def generate_query_dictionary():
    with open("./query_list.tsv") as f:
        '''From Caitlin Westerfield'''
        query_to_word = defaultdict(list)
        for line in f.readlines():
            tokens = line.rstrip().replace(","," ").replace("[", " ").replace("]"," ").replace(">"," ").replace("<"," ").replace("+"," ").replace('"', " ").replace("EXAMPLE_OF("," ").replace(")"," ").split()
            query = tokens[0]
            words = []
            for i in range(1, len(tokens) - 1):
                words.append(re.sub('[A-Za-z]+:','',tokens[i]))
            query_to_word[query] = words
    return query_to_word

def word_to_embedding():
    embedding_dict = defaultdict(list)
    with open("glove.6B.50d.txt") as f:
        for line in f.readlines():
            result = []
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            for i in range(0, 50):
                result.append(float(tokens[i+1]))
            embedding_dict[word] = result
    return embedding_dict

query_to_word = generate_query_dictionary()
word_embedding_dict = word_to_embedding()

def generate_word_embedding(key):
    term_length = len(key)
    total_embed = [0] * 50
    for word in key:
        term_embed = word_embedding_dict[word]
        if len(term_embed) == 50:
            for i in range(0, 50):
                total_embed[i] += term_embed[i]
    #print total_embed, term_length
    #print [x / term_length for x in total_embed]
    return [x / term_length for x in total_embed]

def cleaned_data(file_name, doc_loc, labels, doc_end):
    with open(file_name, "r") as f:
            '''Read in the file'''
            lines = f.readlines()

            '''Initialize the dictionaries for features
            score_dict concatenates the 20 scores for each document
            returned by the query
            length_dict concatenates the length of each of the 20 documents
            normalized by the maximum size of a document (916)
            '''
            score_dict = defaultdict(list)
            length_dict = defaultdict(list)

            for line in lines:
                '''Split the line into tokens'''
                token_list = line.split(" ")
                if token_list[-1] == '\n':
                    token_list = token_list[:-1]

                '''Append the score from the results file'''
                score_dict[token_list[0]].append(token_list[-2])

                '''Get the file name in the docs folder (all xmls)'''
                docname = doc_loc+token_list[2]+doc_end
                doc_size = get_file_length(docname)

                '''Append the document sizes to the proper dicitonary'''
                length_dict[token_list[0]].append(doc_size)

            '''Normalize the length dictionary by the maximum size of a document
            so that the neural network doesn't have to deal with overly large
            numbers'''
            norm_length_dict = normalize_length_dict(length_dict)
            for key in score_dict:
                score_dict[key] += norm_length_dict[key]
                score_dict[key] += generate_word_embedding(query_to_word[key])

    with open(labels, "r") as fp:
            lines = fp.readlines()
            labels_dict = {}
            for line in lines:
                tokens = make_tuple(line)
                label = tokens[0]
                answer = int(tokens[1])
                answer_list = [0] * 20
                answer_list[answer - 1] = 1
                labels_dict[label] = answer_list

    new_dict = {}
    for key in labels_dict:
        if len(score_dict[key]) == FEATURE_SIZE:
            new_dict[key] = (score_dict[key], labels_dict[key])
    return new_dict
