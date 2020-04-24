import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def get_caption(image_path, model_path, vocab_path = 'data/vocab.pkl'):

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    model = torch.load(model_path)
    encoder = model[0]
    decoder = model[1]

    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    y_pred, _ = decoder.beam_search(feature)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in y_pred[1:]:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        else:
            sampled_caption.append(word)
        
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    return sentence

if __name__ == '__main__':
    image_path = "data/train2014/COCO_train2014_000000000064.jpg"
    model_path = "model/best_model (2).pth"
    sentence = get_caption(image_path, model_path, vocab_path = 'data/vocab.pkl')
    print(sentence)