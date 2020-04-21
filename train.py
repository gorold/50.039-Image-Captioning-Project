import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from augmentations import get_augmentations
from train_fasttext import load_fasttext

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    data_augmentations = get_augmentations(args.crop_size)
    
#     glove_path = os.path.join(os.getcwd(),'')
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

#     glove_embed = glove_extract(glove_path)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, data_augmentations, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Load word embeddings
    fasttext_wv = load_fasttext(args.caption_path)
    embed_dim = fasttext_wv.vectors_vocab.shape[1]
    embedding_weights = np.zeros((len(vocab), embed_dim))
    for idx, word in enumerate(vocab.word2idx):
        embedding_weights[idx] = fasttext_wv[word]

    # Build the models
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(args.lstm1_size, args.lstm2_size, args.att_size, vocab, args.embed_size, embedding_weights, feature_size=args.feature_size).to(device)
    # None to change to embedding weights? ^^^
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        encoder.train()
        decoder.train()
        for i, (images, captions, lengths) in enumerate(data_loader):
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward, backward and optimize
            input_lengths = [l-1 for l in lengths]
            features = encoder(images) # The encoder generates the features, which is passed into the LSTM as the first input
            outputs, _ = decoder(features, captions, input_lengths)
            loss = criterion(outputs.transpose(1,2), captions[:,1:])
            
            loss.backward()
            decoder_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/coco2014/train2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/coco2014/trainval_coco2014_captions/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--lstm1_size', type=int, default=512, help='dimension of lstm1 hidden states')
    parser.add_argument('--lstm2_size', type=int, default=512, help='dimension of lstm2 hidden states')
    parser.add_argument('--att_size', type=int, default=256, help='dimension of attension inner dimension')
    parser.add_argument('--feature_size', type=int, default=2048, help='dimension of CNN output')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)