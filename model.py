import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import nltk

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from build_vocab import encode_input_tensor, encode_target_tensor

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab, embedding_size, n_layers, embedding_weights, annoy_objs = None, dropout_p=0.1, temperature = 0.5, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), max_seq_length = 20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.temperature = temperature
        self.device = device
        self.annoy_objs = annoy_objs

        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.softmax = nn.Softmax(dim = 1)
        self.max_seg_length = max_seq_length

        # initialise the embedding layer
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
    
    def init_hidden(self, batch_size = 1):
        # Initialise hidden and cell states with specific batch_size
        # Do not call directly
        init_hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        init_cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        init_hidden_state, init_cell_state = init_hidden_state.to(self.device), init_cell_state.to(self.device)

        return init_hidden_state, init_cell_state

    def word_embed(self, list_of_captions, first_input_tensor):
        '''
        Takes a list of sentences and converts it to input X and ground truth. Word-level embedding using pre-trained word-embedding
        inputs:
            list_of_captions: A list of captions, one caption per batch
            type: list

            first_input_tensor: The reshaped output of the encoder. To be of shape (batch_size, input_size)
            type: tensor
        
        returns:
            packed_input_tensor_batch: A packed sequence for the tensors in a batch. Each tensor in the batch has shape (sentence_length, #characters)
            type: torch.nn.utils.rnn.PackedSequence

            padded_target_tensor_batch: Tensor of shape (batch, max_sequence_length)
            type: tensor
        '''
        input_tensor_batch = []
        target_tensor_batch = []
        if first_input_tensor is not None:
            assert first_input_tensor[1].shape == self.embedding_size, 'First input shape must be equivalent to the embedding shape'
            input_tensor_batch.append(input_tensor_batch)

        for caption in list_of_captions:
            # Pass the caption tensors through the embedding and dropout layers
            input_tensor_batch.append(self.dropout(self.embedding(encode_input_tensor(caption, self.vocab).to(self.device))))
            target_tensor_batch.append(encode_target_tensor(caption, self.vocab, first_input_tensor is not None).to(self.device))

        packed_input_tensor_batch = pack_sequence(input_tensor_batch, enforce_sorted = False)
        padded_target_tensor_batch, sequence_lengths = pad_packed_sequence(pack_sequence(target_tensor_batch, enforce_sorted=False), padding_value=-1, batch_first=True)

        return packed_input_tensor_batch, sequence_lengths, padded_target_tensor_batch
        
    def forward(self, list_of_captions, first_input_tensor = None, encoding_feature_map = None):
        '''
        Standard forward function to train the RNN
        inputs:
            list_of_captions: A list of captions, one caption per batch
            type: list

            first_input_tensor: The reshaped output of the encoder
            type: tensor

            encoding_feature_map: Used for attention layer
            type: tensor

        returns:
            X: Predicted output sequence
            type: tensor

            y: Ground truth output sequence
            type: tensor
        '''
        X, sequence_lengths, y = self.word_embed(list_of_captions, first_input_tensor)
        self.batch_size = sequence_lengths.shape[0] # Get Batch Size
        self.hidden = self.init_hidden(batch_size = self.batch_size) # Initialise hidden and cell states
        
        X = self._internal_forward(X)
        y = y.view(-1)
        return X, y

    def _internal_forward(self, X, sample = False):
        '''
        Forward function used by both training and sampling methods. Not to be called directly
        inputs:
            X: Packed sequence of [input]
            type: torch.nn.utils.rnn.PackedSequence

            sample: boolean flag on whether to sample or not i.e. to test
            type: bool

        returns:
            X: Predicted output sequence
            type: tensor
        '''
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
    
    def _get_softmax_probs(self, X, embed = True):
        if embed:
            X = self.embedding(X)
            X = self.dropout(X)
        X = self._internal_forward(X, sample = True)
        return X

    def beam_search(self, first_input_tensor, top_n = 10, top_k_per_n = 10):
        '''
        Generate a caption using beam search. At evey step, maintain the top 10 sequences. If the next most probable word is the EOS, store the sequence.
        If there are insufficient sequences in the top 10 sequences due to the EOS token, check all completed sequences and return the top most.
        Beam search is determinative. You will not get different results from running this twice.
        inputs:
            first_input_tensor: tensor of hidden state from CNN
            type: tensor

            top_n: int to maintain the n tensors per step
            type: int

            top_k_per_n: int to find the next top k words for each of the n sequences present
            type: int

        returns:
            output_caption: Output Caption of Image
            type: str
        '''
        first_input_tensor = first_input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():  # no need to track history in sampling
            self.hidden = self.init_hidden(batch_size = 1) # Initialise hidden and cell states
            sequences = []
            completed_sequences = []
            for t in range(self.max_seg_length):
                if t == 0:
                    probs = self._get_softmax_probs(first_input_tensor).cpu().numpy().squeeze(0)
                    probs = np.log10(probs)
                    total = 0
                    for word_idx in np.argsort(probs):
                        if total == top_n:
                            break
                        if word_idx != 0:
                            sequences.append((probs[word_idx],[word_idx]))
                            total += 1
                elif t > 0:
                    new_sequences = []
                    for sequence in sequences:
                        probs = self._get_softmax_probs(torch.tensor([sequence[1][-1]]).unsqueeze(0).type(torch.long).to(self.device)).cpu().numpy().squeeze(0)
                        probs = np.log10(probs) # Use log probs to avoid getting 0 for computation
                        for word_idx in np.argsort(probs)[:top_k_per_n]:
                            new_sequence = (sequence[0]+probs[word_idx], sequence[1] + [word_idx])
                            if word_idx != 0:
                                new_sequences.append(new_sequence)
                            else:
                                completed_sequences.append(new_sequence)
                    if len(new_sequences) < top_n:
                        completed_sequences = sorted(completed_sequences, key = lambda x:x[0], reverse=True)
                        break
                    else:
                        new_sequences = sorted(new_sequences, key = lambda x:x[0], reverse=True)
                        sequences = new_sequences[0:top_n]
            else:
                completed_sequences = sorted(sequences, key = lambda x:x[0], reverse=True)

            top_seq = completed_sequences[0][1]
            caption = ' '.join([self.vocab.idx2word[word_idx] for word_idx in top_seq if word_idx != 0])
            return caption

    def sample_caption(self, first_input_tensor):
        '''
        Generate sample sentence by feeding the output of the CNN into the LSTM continuously until the EOS is returned.
        inputs:
            X: Packed sequence of [input]
            type: torch.nn.utils.rnn.PackedSequence

            sample: boolean flag on whether to sample or not i.e. to test
            type: bool

        returns:
            output_caption: Output Caption of Image
            type: str
        '''
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

class DecoderAttentionLSTM(nn.Module):
    def __init__(self):
        super(DecoderAttentionLSTM, self).__init__()
        
    
    def forward(self, feature_map, captions, lengths):
        pass
    
    def sample(self):
        pass
    