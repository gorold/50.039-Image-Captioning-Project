import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import time
import logging

from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from augmentations import get_augmentations
from train_fasttext import load_fasttext
from util.train import train_model, validate_and_plot
from util.metrics import Accuracy, RougeBleuScore
from pycocotools.coco import COCO

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Setup Logger
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    start_time = time.ctime()
    logging.basicConfig(filename= os.path.join(log_file_path,"training_log_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    data_augmentations = get_augmentations(args.crop_size)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    val_coco = COCO(args.val_caption_path)
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loaders
    validation_dataloader = get_loader(args.val_dir, val_coco, vocab, 
                             transform, data_augmentations, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(),
        ])

    validation_dataloader_org = get_loader(args.val_dir, val_coco, vocab, 
                            transform, None, args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Load Model
    encoder, decoder = torch.load(args.model_path, map_location = device)
    
    # Metrics
    metrics = [
        Accuracy(),
        RougeBleuScore(val_coco, vocab)
    ]

    validate_and_plot(validation_dataloader,
                    validation_dataloader_org,
                    val_coco,
                    args.val_dir,
                    encoder,
                    decoder,
                    metrics,
                    top_n = 20,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    logger = None,
                    verbose = True,
                    plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
                    prefix = 'Val')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--train_dir', type=str, default='data/coco2014/train2014', help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='data/coco2014/val2014', help='directory for resized images')
    parser.add_argument('--train_caption_path', type=str, default='data/coco2014/trainval_coco2014_captions/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default='data/coco2014/trainval_coco2014_captions/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--model_path', type=int , default='model/best_model_dl.pth', help='Path to load model')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--lstm1_size', type=int, default=512, help='dimension of lstm1 hidden states')
    parser.add_argument('--lstm2_size', type=int, default=512, help='dimension of lstm2 hidden states')
    parser.add_argument('--att_size', type=int, default=256, help='dimension of attension inner dimension')
    parser.add_argument('--feature_size', type=int, default=1280, help='dimension of CNN output')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)