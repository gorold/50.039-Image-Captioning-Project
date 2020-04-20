import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import nltk

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from build_vocab import encode_input_tensor, encode_target_tensor

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
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]      # Delete the last fc layer and adaptive pooling.
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        """Extract feature maps from input images."""
        features = self.resnet(images)
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
        logits = self.linear(lstm2_out)
        
        return logits, (han, can, hln, cln)        

    def _internal_forward(self, X, sample=False):
        """
        Forward function used by both training and sampling methods. Not to be called directly
        inputs:
            X: Packed sequence of [input]
            type: torch.nn.utils.rnn.PackedSequence

            sample: boolean flag on whether to sample or not i.e. to test
            type: bool

        returns:
            X: Predicted output sequence
            type: tensor
        """
        X, self.hidden = self.lstm(X, self.hidden)
        self.lstm.flatten_parameters() # Save memory

        if not sample:
            X, _ = pad_packed_sequence(X, padding_value=-1, batch_first=True) # Make sure to ignore -1 for cross-entropy function
            X = X.contiguous()

        X = X.view(-1, X.shape[2]) # Reshape to (batch_size * max_seq_length, hidden_size)
        X = self.linear(X) # Output should be shape (batch_size * max_seq_length, # classes)

        if sample:
            X /= self.temperature
            X = self.softmax(X)

        return X
   
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    def _get_softmax_probs(self, X, embed=True):
        if embed:
            X = self.embedding(X)
            X = self.dropout(X)
        X = self._internal_forward(X, sample=True)
        return X
    
    def beam_search(self, features, k=5):
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
        assert features.shape[0] == 1
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
                    softmax_probs = F.softmax(logits, dim=2)
                    top_values, top_indices = softmax_probs.topk(k-len(candidates))
                    for i in range(k - len(candidates)):
                        curr_prob = float(top_values[0,0,i])
                        curr_idx = int(top_indices[0,0,i])             
                        new_sentence = [*sentence] + [curr_idx]
                        new_prob = prob * curr_prob
                        if curr_idx == self.vocab('<end>'):
                            new_final_candidates.append([new_sentence, curr_prob])
                        else:
                            new_temp_candidates.append([new_sentence, hans, cans, hlns, clns, curr_prob])
                if len(new_final_candidates) != 0:
                    candidates += new_final_candidates
                new_temp_candidates.sort(key=lambda x: x[5], reverse=True)
                temp_candidates = new_temp_candidates[:k-len(candidates)]
            candidates.sort(key=lambda x: x[1])
        return candidates[-1]
            

    def sample_caption(self, first_input_tensor):
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
        X = first_input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():  # no need to track history in sampling
            self.hidden = self.init_hidden(batch_size = 1) # Initialise hidden and cell states
            output_caption = ''

            for idx in range(self.max_seg_length):
                if idx == 1:
                    X = self._get_softmax_probs(X, embed = True)
                else:
                    X = self._get_softmax_probs(X, embed = True)
                topi = np.random.choice(a = X.shape[1], size = X.shape[0], p = X.squeeze().cpu().numpy())[0]
                if topi == 0: # EOS Token
                    break
                else:
                    word = self.vocab.idx2word[topi]
                    output_caption += ' ' + word
                X = torch.tensor([topi]).unsqueeze(0).to(self.device).type(torch.long)
            return output_caption
