import gensim
from util.downloadFastText import download_model 
import os
import pickle
from pycocotools.coco import COCO
import nltk
from autocorrect import Speller

def download_fasttext_file(alt_filepath = os.getcwd()):
    fasttext_path = download_model('en', alt_filepath, if_exists='ignore')  # English
    print("Model downloaded")
    return fasttext_path

def load_fasttext_obj(fasttext_path):
    fb_model = gensim.models.fasttext.load_facebook_model(fasttext_path)
    print("FB Model loaded")
    return fb_model

def tokenize_n_process(caption, spell):
    caption = str(caption).lower()
    if spell is not None:
        caption = spell(caption)
    tokens = nltk.tokenize.word_tokenize(caption)
    caption = []
    caption.append('<start>')
    caption.extend([token for token in tokens])
    caption.append('<end>')
    with open(f"{os.getcwd()}/wordEmbeddings/spell_checked_captions.pickle", 'wb') as fp:
        pickle.dump(caption, fp)
    return caption

def train_words(fb_model, json_fp, spell):
    coco_captions_dict = COCO(json_fp).anns

    # Load spell checked captions
    if os.path.isfile(f"{os.getcwd()}/wordEmbeddings/spell_checked_captions.pickle"):
        with open(f"{os.getcwd()}/wordEmbeddings/spell_checked_captions.pickle", 'rb') as fp:
            captions = pickle.load(fp)
    else:
        captions = [tokenize_n_process(i['caption'], spell) for i in coco_captions_dict.values()]

    # Build and train fasttext model
    fb_model.build_vocab(captions, update=True)
    fb_model.train(sentences=captions, total_examples=len(captions), epochs=5)

    # Save word embeddings
    with open(f"{os.getcwd()}/wordEmbeddings/trained_fasttext.pickle", 'wb') as fp:
        pickle.dump(fb_model.wv, fp)
    print('Trained model saved')
    return fb_model

def load_fasttext(json_fp, spell_check = True):
    if os.path.isfile(f"{os.getcwd()}/wordEmbeddings/trained_fasttext.pickle"):
        with open(f"{os.getcwd()}/wordEmbeddings/trained_fasttext.pickle", 'rb') as fp:
            fb_model_wv = pickle.load(fp)
    else:
        fasttext_path = download_fasttext_file()
        fb_model = load_fasttext_obj(fasttext_path)
        if spell_check:
            spell = Speller()
        else:
            spell = None
        fb_model_wv = train_words(fb_model, json_fp, spell)
    return fb_model_wv

if __name__ == "__main__":
    json_fp = f'{os.getcwd()}/data/annotations/captions_train2014.json'
    fb_model_wv = load_fasttext(json_fp)
    # Example
    print(fb_model_wv.wv.similarity('computer', 'human'))
    print(fb_model_wv.wv['hello'])
    print(fb_model_wv.wv.vectors_vocab.shape)