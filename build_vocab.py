import nltk
import pickle
import argparse
import bcolz
import os
import numpy as np
import annoy
import lmdb
import sys
import torch
import re
import unicodedata

from tqdm import tqdm
from collections import Counter
from pycocotools.coco import COCO

class iteratefromdict():
    def __init__(self, cat_dict, train, batch_size, seed, test = False, percentage_training = 0.8, percentage_test = 0.1):
        self.train = train
        self.seed = seed
        self.cat_dict = cat_dict
        self.ct = 0
        self.batch_size = batch_size
        self.namlab = cat_dict

        np.random.seed(self.seed)
        np.random.shuffle(self.namlab)
        if train:
            self.namlab = self.namlab[:int(percentage_training * len(self.namlab))]
        elif test:
            self.namlab = self.namlab[int(percentage_training * len(self.namlab)):int((percentage_training + percentage_test) * len(self.namlab))]
        else:
            self.namlab = self.namlab[int((percentage_training + percentage_test) * len(self.namlab)):]
            
    def num(self):
        return len(self.namlab)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ct >= len(self.namlab):
            self.ct = 0
            raise StopIteration()
        else:
            self.ct += self.batch_size
            if self.ct >= len(self.namlab): # If overflow from list, then get from the start
                remainder = self.batch_size - len(self.namlab[self.ct-self.batch_size:])
                return self.namlab[self.ct-self.batch_size:] + self.namlab[:remainder]
            else:
                return self.namlab[self.ct - self.batch_size : self.ct]

    def __len__(self):
        return int(len(self.namlab)/self.batch_size)

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    with tqdm(ids, desc='Tokenizer', file=sys.stdout, disable=False) as iterator:
        for id in iterator:
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(normalize_string(caption.lower()))
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<EOS>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def get_all_captions(json):
    coco = COCO(json)
    return [normalize_string(val['caption']) for val in coco.anns.values()]

def create_annoy_index(file_dir, glove, num_trees=30, verbose=True):
    '''
    Create annoy index for GLOVE vectors
    Create 2 way dictionary and store in lmdb
    '''
    fn_annoy = os.path.join(file_dir,'annoy_idx.annoy')
    fn_lmdb = os.path.join(file_dir,'word_idx_dict.lmdb') # stores word <-> id mapping

    vec_length = list(glove.values())[0].shape[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    env = lmdb.open(fn_lmdb, map_size=int(1e9))
    if not os.path.exists(fn_annoy) or not os.path.exists(fn_lmdb):
        annoy_index = annoy.AnnoyIndex(vec_length, metric='angular')
        with env.begin(write=True) as txn:
            with tqdm(enumerate(glove.keys()), desc='Creating Index', file=sys.stdout, total = len(glove)) as iterator:
                for idx, word in iterator:
                    vec = glove[word]
                    annoy_index.add_item(idx, vec)
                    word_id = 'i%d' % idx
                    word = 'w' + word
                    txn.put(word_id.encode(), word.encode())
                    txn.put(word.encode(), word_id.encode())
        if verbose:
            print("Starting to build")
        annoy_index.build(num_trees)
        if verbose:
            print("Finished building")
        annoy_index.save(fn_annoy)
        if verbose:
            print(f"Annoy index saved to: {fn_annoy}")
            print(f"lmdb map saved to: {fn_lmdb}")
    else:
        print("Annoy index and lmdb map already in path")

def glove_extract(glove_path):
    '''
    Extract GLOVE vectors from .dat and stores in a pickle file for easy access
    GLOVE Extraction code: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    ''' 
    if not os.path.exists(os.path.join(glove_path,'glove.6B.50d.dat')) or not os.path.exists(os.path.join(glove_path,'glove.6B.50d.words.pk')) or not os.path.exists(os.path.join(glove_path,'glove.6B.50d.word2idx.pk')):
        words = []
        idx = 0
        word2idx = {}
        
        vectors = bcolz.carray(np.zeros(1), rootdir=os.path.join(glove_path,'glove.6B.50d.dat'), mode='w')

        with open(os.path.join(glove_path,'glove.6B.50d.txt'), 'rb') as f:
            with tqdm(f, desc='Extracting Glove', file=sys.stdout, total = 400000) as iterator:
                for l in iterator:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1
                    vect = np.append(np.array(line[1:]).astype(np.float),0.)
                    vectors.append(vect)
            shape = vect.shape[0]
            
        vectors = bcolz.carray(vectors[1:].reshape((400000, shape)), rootdir=os.path.join(glove_path,'glove.6B.50d.dat'), mode='w')
        vectors.flush()
        pickle.dump(words, open(os.path.join(glove_path,'glove.6B.50d.words.pk'), 'wb'))
        pickle.dump(word2idx, open(os.path.join(glove_path,'glove.6B.50d.word2idx.pk'), 'wb'))
    else:
        vectors = bcolz.open(os.path.join(glove_path,'glove.6B.50d.dat'))[:]
        words = pickle.load(open(os.path.join(glove_path,'glove.6B.50d.words.pk'), 'rb'))
        word2idx = pickle.load(open(os.path.join(glove_path,'glove.6B.50d.word2idx.pk'), 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

def glove_decode(vector_query, annoy_index, key_idx_db, num_results = 1, verbose=False):
    ''' Returns the closest K words given the vector query
    '''
    ret_keys = []
    with key_idx_db.begin() as txn:
        for id in annoy_index.get_nns_by_vector(vector_query, num_results):
            key = txn.get(('i%d' % id).encode())[1:]
            ret_keys.append(key.decode())
    if verbose:
        print(f"Found: {len(ret_keys)} results")
    return ret_keys

def encode_input_tensor(caption, vocab):
    '''
    Takes in a caption and returns the index corresponding to that caption
    '''
    caption_tensor = []
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    for word in tokens:
        caption_tensor.append(vocab.word2idx[word])
    caption_tensor = torch.tensor(caption_tensor).type(torch.long)
    return caption_tensor

def encode_target_tensor(caption, vocab, captioning = False):
    '''
    Takes in a caption and returns the target tensor corresponding to that caption
    '''
    caption_tensor = []
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    for word in tokens:
        caption_tensor.append(vocab.word2idx[word])
    caption_tensor.append(0) # 0 is the index for the EOS token
    caption_tensor = torch.tensor(caption_tensor).type(torch.long)
    if captioning:
        return caption_tensor
    else:
        return caption_tensor[1:]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=0, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)