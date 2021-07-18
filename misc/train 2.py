import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
from model import EncoderFeedforward, EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import *
from gen_sequence import *


if not os.path.exists("plots/plots_cnn"):
    os.mkdir("plots/plots_cnn")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = ['blur', 'blend', 'translate', 'rotate', 'affine', 'perspective', 'pad', '<start>', '<end>']

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Build the models, can use a feedforward/convolutional encoder and an RNN decoder
    encoder = EncoderCNN(args.embed_size).to(device)    #can be sequential or convolutional
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    # Loss and optimizer
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()
    softmax = nn.LogSoftmax(dim=1)
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    total_training_steps = args.num_iters
    losses = []
    perplexity = []
    for epoch in range(args.num_epochs):
        for i in range(total_training_steps):
            prog_data = generate_training_data(args.batch_size)

            images = [im[0] for im in prog_data]
            transforms = [transform[1] for transform in prog_data]

            [ele.insert(0,'<start>') for ele in transforms]                  #start token for each sequence
            [ele.append('<end>') for ele in transforms]                      #end token for each sequence
                
            lengths = [len(trans) for trans in transforms]

            maximum_len = max(lengths)
            for trans in transforms:
                if len(trans) != maximum_len:
                    trans.extend(['pad']*(maximum_len-len(trans)))

            padded_lengths = [len(trans) for trans in transforms]
            transforms = [[word_to_int(word) for word in transform] for transform in transforms]
            transforms = torch.tensor(transforms, device=device)
            images = torch.tensor(images,device=device)
            images = images.unsqueeze(1)                                   #Uncomment this line when training using EncoderCNN
            lengths = torch.tensor(lengths,device=device)
            padded_lengths = torch.tensor(padded_lengths,device=device)
            #print(pack_sequence(transforms))
            targets = pack_padded_sequence(transforms, padded_lengths, batch_first=True)[0] 
            #targets = pack_sequence(transforms)[0]
            #print(targets)
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, transforms, padded_lengths)
            #print(outputs)
            
            loss = criterion1(outputs, targets)
            losses.append(loss.item())
            perplexity.append(np.exp(loss.item()))

            #nll_loss = criterion2(softmax(outputs),targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},Perplexity: {:5.4f}'.format(epoch, args.num_epochs, i, total_training_steps, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

    #fig = go.Figure()
    #fig.add_trace(go.Scatter(
    #x=np.arange(len(losses)),
    #y=losses,
    #name = 'Cross Entropy Loss',
    #connectgaps=True 
    #))
    #fig.add_trace(go.Scatter(
    #x=np.arange(len(perplexity)),
    #y=perplexity,
    #name='Perplexity',
    #connectgaps=True 
    #))
    #fig.update_layout(title='Training Accuracy and Perplexity for Model',
    #               xaxis_title='Iterations',
    #               yaxis_title='Cross Entropy Loss and Perplexity')
    #fig.show()
    #fig.write_image("plots/plots_ffwd/ffwd2.png")
    y = losses
    z = perplexity
    x = np.arange(len(losses))
    plt.plot(x, y, label='Cross Entropy Loss')
    plt.plot(x, z, label='Perplexity')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy Loss and Perplexity')
    plt.title("Cross Entropy Loss and Model Perplexity During Training")
    plt.legend()
    plt.savefig('plots/plots_cnn/cnn5_gpu', dpi=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/model5/' , help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
