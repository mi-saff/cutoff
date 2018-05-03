import os
import json
import argparse
from collections import OrderedDict
from collections import defaultdict
#import torch
import re
#from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
from subprocess import call
import subprocess
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--input", type=str, default=-1, help="result file")
parser.add_argument("--train", type=str, default=-1, help="result file")
parser.add_argument("--judg", type=str, default=-1, help="judgment file")

# parse parameters
params = parser.parse_args()

# check parameters
#X = [[0, 0], [1, 1],[0.5,0.1]]
#y = [0, 1,2]
q=defaultdict(list)
qid2cutoff={}
with open(params.train) as f:
    for line in f.readlines():
        result = []
        tokens = line.rstrip().split(" ")
        result.append(tokens[2])
        result.append(tokens[3])
        result.append(tokens[4]) # document, rank, score
        query = tokens[0]
        q[query].append(result)

cutoff = [i+1 for i in range(15)]

X = []
y=[]
n = 50
clf = svm.SVC()
for i in q:
    AQWV_scores = []
    xi = []
    for doc in q[i]:
        xij = doc[2] # indri score
        xi.append(float(xij))
        if len(xi)>=n:
            break
    while(len(xi)<n):
        xi.append(-100.)
    #print(i,len(xi))
    X.append(xi)
    for c in cutoff:
        out = open("query.tsv",'w')
        r = 0
        for doc in q[i]:
            if r > c:
                break
            out.write(i + " Q0 ")
            for j in range(len(doc)):
                out.write(doc[j]+" ")
            out.write("indri\n")
            r +=1
        out.close
        with open("eval.sh", 'w') as f:
            f.write("trec_eval -q "+params.judg+" -m aqwv query.tsv > predict.tsv")
        f.close()
        call(["sh","eval.sh"])

        with open("predict.tsv") as f:
            for line in f.readlines(): #it's possible to have empty file
                aqwv = float(line.split("\t")[2])
                AQWV_scores.append(aqwv)
                break

        os.remove("eval.sh")
    best_cutoff = np.argmax(AQWV_scores)+1
    y.append(best_cutoff)
    print(i,best_cutoff)
#print(X,y)
clf.fit(X, y)
q_test_features=defaultdict(list)
q_test_docs=defaultdict(list)
with open(params.input) as f:
    for line in f.readlines():
        result = []
        tokens = line.rstrip().split(" ")
        result.append(tokens[2])
        result.append(tokens[3])
        result.append(tokens[4]) # document, rank, score
        query = tokens[0]
        q_test_features[query].append(float(tokens[4]))
        q_test_docs[query].append(result)

out = open(params.input+".optimized",'w')
for query in q_test_features:
    xi=q_test_features[query]
    while(len(xi)<n):
        xi.append(-100.)
    best_cutof = clf.predict(X)
    r = 0
    for doc in q_test_docs[query]:
        if r > best_cutoff:
            break
        out.write(query + " Q0 ")
        for j in range(len(doc)):
            out.write(doc[j]+" ")
        out.write("indri\n")
        r +=1
        out.close

f.close()

