import torch
import re
import sys, os

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# Functional
def correct_counts(y_pred, y):
    y_pred, y = y_pred.cpu(), y.cpu()
    mask = (y > 0)
    _, indices = torch.max(y_pred, 1)
    return (indices[mask] == y[mask]).float()

def tensor_to_words(y_pred, y, vocab):
    y_pred, y = y_pred.cpu(), y.cpu()
    mask = (y > 0)
    _, indices = torch.max(y_pred, 1)

    predicted_words_idx = list(indices.cpu().numpy())
    masks = list(mask.cpu().numpy())
    captions = [caption[masks[i]] for i, caption in enumerate(predicted_words_idx)]
    for idx, caption in enumerate(captions):
        captions[idx] = ' '.join([vocab.idx2word[i] for i in caption])
    return captions

def extract_captions(image_ids, caption_pred_list, coco):
    gt_captions = {}
    pred_captions = {}
    for idx, id in enumerate(image_ids):
        gt_captions[id] = [coco.anns[i]['caption'] for i in coco.getAnnIds(id)]
        pred_captions[id] = [caption_pred_list[idx]]
    return pred_captions, gt_captions

# Modules
class Metric(object):
    def __init__(self, name=None):
        self._name = name
    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name 

class Accuracy(Metric):
    __name__ = 'accuracy'

    def __init__(self):
        pass

    def evaluate(self, y_pred, y):
        return correct_counts(y_pred, y)

class RougeBleuScore(Metric):

    def __init__(self, coco, vocab, n = 4):
        self.coco = coco
        self.vocab = vocab
        self.bleu = Bleu(n)
        self.n = n
        self.rouge = Rouge()

    def evaluate(self, y_pred, y, image_ids):
        caption_pred_list = tensor_to_words(y_pred, y, self.vocab)
        captions_pred, captions_gt = extract_captions(image_ids, caption_pred_list, self.coco)
        blockPrint()
        scores = self.bleu.compute_score(captions_gt, captions_pred)[0]
        enablePrint()
        scores.append(self.rouge.compute_score(captions_gt, captions_pred)[0])
        return scores