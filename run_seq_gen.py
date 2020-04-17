import argparse
import os
import logging
import time
import torch
import annoy
import lmdb
import pickle
import numpy as np

from data_loader import get_loader
from torchvision import transforms
from build_vocab import iteratefromdict, get_all_captions, glove_extract, create_annoy_index, Vocabulary
from utils.misc import log_print
from utils.train import train_model
from model import DecoderRNN

def main(args):
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    start_time = time.ctime()
    logging.basicConfig(filename= os.path.join(log_file_path,"training_log_classifier_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    # Get Captions and GLOVE, create ANNOY index
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    log_print(f'Vocab Loaded', logging)
    captions = get_all_captions(args.caption_path)[0:10000]
    log_print(f'Captions Loaded', logging)
    glove = glove_extract(args.glove_path)
    log_print(f'GLOVE Loaded', logging)
    # create_annoy_index(args.annoy_dir, glove)
    # log_print(f'ANNOY Index Created', logging)
    embed_dim = list(glove.values())[0].shape[0]
    # annoy_objs = dict(
    #     annoy_index = annoy.AnnoyIndex(embed_dim, metric='angular').load(os.path.join(args.annoy_dir,'annoy_idx.annoy')),
    #     key_idx_db = lmdb.open(os.path.join(args.annoy_dir,'word_idx_dict.lmdb'), map_size=int(1e9))
    # )

    # Prepare dataloaders
    train_dataloader = iteratefromdict(captions, train = True, seed = 5, batch_size = args.batch_size)
    val_dataloader = iteratefromdict(captions, train = False, seed = 5, batch_size = args.batch_size)
    test_dataloader = iteratefromdict(captions, train= False, test = True, seed = 5, batch_size = args.batch_size)

    # Setup the Weight Matrix for Embedding Layer
    embedding_weights = np.zeros((len(vocab), embed_dim))
    words_not_found = 0
    for idx, word in enumerate(vocab.word2idx):
        if word in glove.keys():
            embedding_weights[idx] = glove[word]
        elif word == '<EOS>':
            embedding_weights[idx] = np.append(np.zeros(embed_dim-1),1)
        else:
            embedding_weights[idx] = np.random.normal(scale=0.6, size=(embed_dim, ))
            words_not_found += 1
    
    log_print(f'{words_not_found} words not found in GLOVE', logging)
    
    decoder = DecoderRNN(hidden_size = args.hidden_size,
                        vocab = vocab,
                        embedding_size = embed_dim,
                        n_layers = args.num_layers,
                        embedding_weights = embedding_weights,
                        temperature = 0.5)

    loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #Ignore padded index
    learning_rate = 0.001
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0)

    train_model(train_dataloader = train_dataloader,
                validation_dataloader = val_dataloader,
                model = decoder,
                loss = loss,
                optimizer = optimizer,
                scheduler = scheduler,
                batch_size = args.batch_size,
                num_epochs = args.num_epochs,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = logging,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'model'),
                plots_save_path = os.path.join(os.getcwd(),'plots')
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pk', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--glove_path', type=str, default='data/glove/', help='path for glove text file')
    parser.add_argument('--annoy_dir', type=str, default='data/annoy/', help='dir for annoy index and lmdb mutual key dict')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int , default=300, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)