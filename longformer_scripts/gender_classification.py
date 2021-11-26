import json
import numpy as np
import argparse
from collections import OrderedDict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
from sklearn.metrics import confusion_matrix

gender_dict = {'male':0, 'female':1, 'neutral':2}


def read_json(path):
    with open(path,'r') as f:
        data = json.load(f)
        return OrderedDict(data)
    
def read_dict(path):
    idx = 4
    voc = {2:'<EOS>',3:'<unk>'}
    with open(path,'r') as f:
        for line in f.readlines():
            token,freq = line.split()
            voc[idx] = token
            idx += 1
        return voc


def read_labels(labels_file):
    labels = {}
    positions = {}
    professions = {}
    idx = 0
    with open(labels_file) as lf:
        for line in lf.readlines():
            line = line.split('\t')
            gender = gender_dict[line[0]]
            pos = int(line[1])
            labels[idx] = gender
            positions[idx] = pos
            professions[idx] = line[2].split()[pos]
            idx += 1
        return labels, positions, professions

def match_embeddings(source,voc):
    indices = []
    idx = 0
    if isinstance(source[0],list):
        source = source[0]  
    token_source = [voc[s] for s in source]
    for token in token_source:
        indices.append(idx)
        if '@@'  not in token:
            idx += 1
    return indices


def average_subword(encodings, token_positions,pos,  idx):
    res = np.asarray(encodings[idx])
    n = 1
    while (token_positions[idx + n] == pos):
        res += np.asarray(encodings[idx + n]) 
        n += 1
    return res/n

def first_subword(encodings, token_positions,pos,  idx):
    if len(encodings) == 1:
        encodings = encodings[0]
    res = np.asarray(encodings[idx])
    return res
    
def last_subword(encodings, token_positions,pos,  idx):
    res = np.asarray(encodings[idx])
    n = 1
    while (token_positions[idx + n] == pos):
        res = np.asarray(encodings[idx + n]) 
        n += 1
    return res
 
def read_embeddings(embeddings_file,vocabulary_file, positions, offset):
    det_embeddings = {}
    prof_embeddings = {}
    data = read_json(embeddings_file)
    voc = read_dict(vocabulary_file)
    for i,dsent in data.items():
        pos = positions[int(i)] + int(offset) 
        token_positions = match_embeddings(dsent['src'],voc)
        idx = token_positions.index(pos)
        prof_embeddings[int(i)] = first_subword(dsent['encoding'],token_positions, pos, idx)
        det_embeddings[int(i)] =  first_subword(dsent['encoding'],token_positions, pos, idx-1)
    return prof_embeddings, det_embeddings

def split_data(prof_embeddings, det_embeddings, labels, prof_names, train_size=1000):
    indexes = random.sample(list(range(len(prof_embeddings))), train_size)
    mask = np.ones(len(labels), dtype=bool)
    mask[indexes] = False

    det_np = np.asarray([det_embeddings[i] for i in range((len(det_embeddings)))]) 
    prof_np = np.asarray([prof_embeddings[i] for i in range((len(prof_embeddings)))])      
    labels_np = np.asarray([labels[i] for i in range((len(labels)))])
        
    det_train = det_np[indexes]
    prof_train = prof_np[indexes]
    lab_train =  labels_np[indexes]
    det_test = det_np[mask]
    prof_test = prof_np[mask]
    lab_test = labels_np[mask]

    prof_names_test = [prof_names[i] for i in range(len(prof_names)) if not i in indexes] 

    return det_train, prof_train, lab_train,det_test, prof_test, lab_test, prof_names_test


def compute_common_errors(prof_names, preds, test_y, n=50):
    error_freq = {}
    idx = 0
    for p,y,name in zip(preds, test_y, prof_names):
        if not p == y:
            if name in error_freq: 
                error_freq[name] += 1 
            else:
                error_freq[name] = 1
    
    error_list = list(error_freq.items())
    error_list = sorted(error_list, key= lambda x: x[1], reverse=True)
    print('Most common errors')
    for i in range(1,n+1):
        print(i, error_list[i-1])

def train_svm(train_x, train_y, test_x, test_y, prof_names):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    print('Accuracy', clf.score(test_x,test_y))
    print('Confusion_matrix\n', confusion_matrix(test_y,pred))
    compute_common_errors(prof_names, pred, test_y)
    


parser = argparse.ArgumentParser(description='Train Gender classifier from contextual embeddings')
parser.add_argument('-e', '--embeddings',  help='Embeddings on json format', required=True)
parser.add_argument('-l', '--labels', help='Text file containing the labels', required=True)
parser.add_argument('-v', '--vocabulary', help='Text file containing the vocabulary', required=True)
parser.add_argument('-o', '--offset', help='Offset for added characters to the sentence', required=True)
parser.add_argument('-s', '--seed', help='Seed for random generator', required=True)
args = parser.parse_args()

random.seed(int(args.seed))
labels, positions, professions = read_labels(args.labels)
prof_embeddings, det_embeddings = read_embeddings(args.embeddings, args.vocabulary, positions, args.offset)
det_train, prof_train, labels_train, det_test, prof_test, labels_test, prof_test_names = split_data(prof_embeddings, 
                                                                                                det_embeddings, 
                                                                                                labels, 
                                                                                                professions)
print('* Determiners:')
train_svm(prof_train, labels_train, prof_test, labels_test, prof_test_names)
print('* Profession:')
train_svm(det_train, labels_train, det_test, labels_test, prof_test_names)


