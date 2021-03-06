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
                             transform, None, 1,
                             shuffle=True, num_workers=args.num_workers, limit = 1000)

    # Load Model
    encoder, decoder = torch.load(args.saved_model_path, map_location = device)
    
    # Metrics
    metrics = [
        Accuracy(),
        RougeBleuScore(val_coco, vocab)
    ]

    validate_and_plot(validation_dataloader,
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
    parser.add_argument('--val_dir', type=str, default='data/val2014', help='directory for resized images')
    parser.add_argument('--val_caption_path', type=str, default='data/annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--saved_model_path', type=str , default='model/best_model_en.pth', help='Path to load model')
    
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()
    main(args)