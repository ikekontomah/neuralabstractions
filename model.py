import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
from gen_sequence import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class EncoderFeedforward(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderFeedforward, self).__init__()
        self.layer1 = nn.Linear(64*64, embed_size*2)
        self.layer2 = nn.Linear(embed_size*2,embed_size*2)
        self.layer3 = nn.Linear(embed_size*2,embed_size*2)
        self.out = nn.Linear(embed_size*2, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        images = images.float()
        features = images.reshape(images.size(0),-1)
        #features = torch.nn.functional.relu(features)
        features = self.bn(torch.nn.functional.relu(self.out(self.layer3(self.layer2(self.layer1(features))))))
        return features

#EncoderCNN model using a CNN instead of sequential layer
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.drop_out = nn.Dropout()
        self.fc = nn.Linear(256 * 4 * 4, embed_size)
        self.out = nn.Linear(embed_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            images = images.float()
            features = self.layer1(images)
            features = self.layer2(features)
            features = self.layer3(features)
            features = self.layer4(features)
        features = features.reshape(features.size(0), -1)
        features = self.drop_out(features)
        features = self.fc(features)
        features = self.out(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=max_len+2):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        #packed = pack_sequence(embeddings)
        #import pdb; pdb.set_trace()
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        #sampled_ids.append(torch.tensor([0]))
        inputs = features.unsqueeze(1)       #make sure it is the same as
        #features.unsqueeze(1) #torch.cat((features.unsqueeze(1), embeddings), 1)
        #print(inputs.shape)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            #print(sampled_ids)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            #print(inputs)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            #print(inputs)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
