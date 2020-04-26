import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import nltk

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from build_vocab import encode_input_tensor, encode_target_tensor
from efficientnet_pytorch import EfficientNet

class Attention(nn.Module):
    """Additive Attention"""
    def __init__(self, hidden_size, query_size, key_size):
        super(Attention, self).__init__()
        self.v = nn.Linear(hidden_size, 1)
        self.W = nn.Linear(query_size + key_size, hidden_size)
        
    def forward(self, query, keys, values):
        """
        Parameters
        ----------
        query : tensor 
            [B,Q,Nq]
        keys : tensor
            [B,KV,Nk]
        values : tensor
            [B,KV,Nv]
        """
        
        query = query.unsqueeze(dim=1).repeat(1, keys.shape[1],1, 1) # [B,KV,Q,Nq]
        keys = keys.unsqueeze(dim=2).repeat(1, 1, query.shape[2], 1) # [B,KV,Q,Nk]
        feats = torch.cat((query, keys), dim=3) # [B,KV,Q,Nq+Nk]
        energy = torch.tanh(self.W(feats)) # [B,KV,Q,Nq+Nk]
        energy = self.v(energy).squeeze(dim=-1) # [B,KV,Q]
        scores = F.softmax(energy, dim=1) # [B,KV,Q]
        output = torch.bmm(values.transpose(1,2), scores) # [B,Nv,Q]
        output = output.transpose(1,2) # [B,Q,Nv]
        
        return output    

class EncoderCNN(nn.Module):
    def __init__(self, encoder_name = 'efficientnet-b0'):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        if 'efficientnet' in encoder_name:
            net = EfficientNet.from_pretrained('efficientnet-b0')
            net._avg_pooling = nn.Identity()
            net._dropout = nn.Identity()
            net._fc = nn.Identity()
        elif encoder_name == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-2]      # Delete the last fc layer and adaptive pooling.
            net = nn.Sequential(*modules)
        elif encoder_name == 'mobilenet_v2':
            net = models.mobilenet_v2(pretrained=True).features
        self.net = net
        self.encoder_name = encoder_name
        
    def forward(self, images):
        """Extract feature maps from input images."""
        if 'efficientnet' in self.encoder_name:
            features = self.net(images).view(-1,1280,7,7)
        else:
            features = self.net(images)
        features = F.normalize(features, p=2, dim=1) # L2 normalization over channels
        return features

class DecoderRNN(nn.Module):
    def __init__(
        self, 
        lstm1_hidden_size,
        lstm2_hidden_size,
        att_hidden_size,
        vocab, 
        embedding_size, 
        embedding_weights, 
        feature_size=2048,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
        ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = embedding_size
        self.device = device

        lstm1_input_size = embedding_size + feature_size
        lstm2_input_size = feature_size + lstm1_hidden_size

        # LSTM layers
        self.lstm1 = nn.LSTM(lstm1_input_size, lstm1_hidden_size, 1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm2_input_size, lstm2_hidden_size, 1, batch_first=True)

        # Linear layers for Attention Mechanism
        self.attention = Attention(att_hidden_size, lstm1_hidden_size, feature_size)

        # Final prediction layers
        self.linear = nn.Linear(lstm1_hidden_size, len(vocab))

        # initialise the embedding layer
        self.embed = nn.Embedding(len(vocab), embedding_size)
        if embedding_weights is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding_weights))
    
    def forward(self, features, captions, lengths, ha0=None, ca0=None, hl0=None, cl0=None):
        """
        Parameters
        ----------
        features : tensor
            Output of CNN Encoder
        captions : tensor
            Tensor of integers, representing index of words
        """
        
        # Attention-LSTM
        mean_pooled_features = F.adaptive_avg_pool2d(features, (1,1)).squeeze(-1).transpose(1,2)
        embeddings = self.embed(captions)
        mean_pooled_features = mean_pooled_features.repeat(1, embeddings.shape[1], 1)
        lstm1_inp = torch.cat((embeddings, mean_pooled_features), dim=2)
        packed = pack_padded_sequence(lstm1_inp, lengths, batch_first=True)
        if ha0 is None or ca0 is None:
            lstm1_out, (han, can) = self.lstm1(packed)
        else:
            lstm1_out, (han, can) = self.lstm1(packed, (ha0, ca0))
        
        # Attention
        lstm1_out = pad_packed_sequence(lstm1_out, batch_first=True)[0]
        features = features.view(features.shape[0], features.shape[1], -1).transpose(1,2)
        att_out = self.attention(lstm1_out, features, features)
        
        # Language-LSTM
        lstm2_inp = torch.cat((lstm1_out, att_out), dim=2)
        packed = pack_padded_sequence(lstm2_inp, lengths, batch_first=True)
        if hl0 is None or cl0 is None:
            lstm2_out, (hln, cln) = self.lstm2(packed)
        else:
            lstm2_out, (hln, cln) = self.lstm2(packed, (hl0, cl0))
        
#         # Predict Words
        lstm2_out = pad_packed_sequence(lstm2_out, batch_first=True)[0]
        logits = self.linear(lstm2_out).transpose(1,2)
        
        return logits, (han, can, hln, cln)        
   
    def sample(self, features, states=None):
        """
        Generate captions for given image features using greedy search.
        
        Parameters
        ----------
        features : tensor
            Output of EncoderCNN. Tensor of shape [B, C, H, W]
            
        Returns
        -------
        tensor (long)
        """
        batch_size = features.shape[0]
        lengths = [1 for _ in range(batch_size)]
        sentences = torch.empty((batch_size, 0), dtype=torch.long).to(self.device)
        outputs = [False for _ in range(batch_size)]
        done = torch.full((batch_size, 1), False, dtype=torch.bool).to(self.device)
        ha0 = ca0 = hl0 = cl0 = None
        prev_word = torch.full((batch_size, 1), self.vocab('<start>'), dtype=torch.long).to(self.device)
        while not torch.all(done):
            logits, (ha0, ca0, hl0, cl0) = self(features, prev_word, lengths, ha0, ca0, hl0, cl0)
            prev_word = logits.argmax(dim=1)
            sentences = torch.cat((sentences, prev_word), dim=1)
            mask = prev_word == self.vocab('<end>')
            if torch.any(mask):
                for idx in mask.nonzero():
                    idx = idx[0]
                    if done[idx] != True:
                        outputs[idx] = sentences[idx]
                done[mask] = True
        return outputs
    
    def beam_search(self, features, k=5, max_seq_length = 200):
        """
        Generate a caption using beam search. At evey step, maintain the top k candidates. If the next most probable word is the EOS, store the sequence.
        Beam search is deterministic. You will not get different results from running this twice.
        
        Parameters
        ----------
        features : tensor
            Output of EncoderCNN (batch size of 1). tensor of shape (1, C, H, W)

        k : int
            Beam search parameter, number of candidates to consider

        Returns
        -------
        str
            Candidate sentence with highest joint probability
        """
        assert features.shape[0] == 1, 'Batch size has to be 1'
        self.eval()
        candidates = []
        temp_candidates = [[[self.vocab('<start>')], None, None, None, None, 1]]
        with torch.no_grad():
            while len(candidates) < k:
                new_temp_candidates = []
                new_final_candidates = []
                for sentence, ha0, ca0, hl0, cl0, prob in temp_candidates:
                    prev_word = torch.zeros(1,1).long().to(self.device)
                    prev_word[0,0] = sentence[-1]
                    lengths = [1]
                    logits, (hans, cans, hlns, clns) = self(features, prev_word, lengths, ha0, ca0, hl0, cl0)
                    logits = logits.permute(0,2,1)
                    softmax_probs = F.softmax(logits, dim=2)
                    top_values, top_indices = softmax_probs.topk(k-len(candidates))
                    for i in range(k - len(candidates)):
                        curr_prob = float(top_values[0,0,i])
                        curr_idx = int(top_indices[0,0,i])             
                        new_sentence = [*sentence] + [curr_idx]
                        new_prob = prob * curr_prob
                        if curr_idx == self.vocab('<end>') or len(new_sentence) >= max_seq_length:
                            new_final_candidates.append([new_sentence, curr_prob])
                        else:
                            new_temp_candidates.append([new_sentence, hans, cans, hlns, clns, curr_prob])
                if len(new_final_candidates) != 0:
                    candidates += new_final_candidates
                new_temp_candidates.sort(key=lambda x: x[5], reverse=True)
                temp_candidates = new_temp_candidates[:k-len(candidates)]
            candidates.sort(key=lambda x: x[1])
        return candidates[-1]

    def sample_caption(self, features):
        """
        Generate sample sentence by feeding the output of the CNN into the LSTM continuously until the EOS is returned.
        inputs:
            X: Packed sequence of [input]
            type: torch.nn.utils.rnn.PackedSequence

            sample: boolean flag on whether to sample or not i.e. to test
            type: bool

        returns:
            output_caption: Output Caption of Image
            type: str
        """
        assert features.shape[0] == 1, 'Batch size has to be 1'
        self.eval()
        candidates = []
        sentence, hans, cans, hlns, clns = [self.vocab('<start>')], None, None, None, None
        with torch.no_grad():
            for idx in range(200):
                prev_word = torch.zeros(1,1).long().to(self.device)
                prev_word[0,0] = sentence[-1]
                lengths = [1]
                logits, (hans, cans, hlns, clns) = self(features, prev_word, lengths, hans, cans, hlns, clns)
                softmax_probs = F.softmax(logits, dim=1)
                next_word_idx = np.random.choice(a = len(self.vocab), size = 1, p = softmax_probs.squeeze().cpu().numpy()).item()
                sentence = [*sentence] + [next_word_idx]
                if next_word_idx == self.vocab('<end>'):
                    break
        sentence = ' '.join([self.vocab.idx2word[idx] for idx in sentence[1:]])
        return sentence
