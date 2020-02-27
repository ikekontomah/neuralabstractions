import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from model import EncoderFeedforward, EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import *
from gen_sequence import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = ['blur', 'translate', 'rotate', 'affine', 'perspective', 'pad', '<start>', '<end>']

def main(args):

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))    #path for dumbing output of encoder model
    decoder.load_state_dict(torch.load(args.decoder_path))   #path for dumbing output of decoder model

    test_data = generate_training_data(5)
    textures_test = generate_textures(test_data)
    transforms_test = generate_transforms(test_data)
    for i in range(len(textures_test)):
        plt.imsave('predictions/texture4_0%i.png'%i, textures_test[i], cmap="gray")
        
    print(transforms_test)
    predicted_progs = []

    for texture in textures_test:
        texture = torch.tensor(texture, device=device)
        texture = texture.unsqueeze(0)        
        texture = texture.unsqueeze(0)                      #for EncoderCNN ought to unsqueeze twice
        feature = encoder(texture)
        sampled_seq = decoder.sample(feature)
        sampled_seq = sampled_seq[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        # Convert sampled sequence of transforms to words
        prog = []
        for int_word in sampled_seq:
            word = int_to_word(int_word)
            prog.append(word)
            if word == '<end>':
                break
        trans_seq = '-->'.join(prog)
        predicted_progs.append([trans_seq])
        
    # Print out the sequence of generated transform sequences
    print(predicted_progs)
        

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/model4/encoder-5-4000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/model4/decoder-5-4000.ckpt', help='path for trained decoder')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
