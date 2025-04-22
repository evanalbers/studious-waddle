""" small rnn to learn shakepeare, learn activations from """
import torch.nn as nn
import torch

class rnn(nn.Module): 
    """ rnn to predict next character on given input text """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, num_layers=3):
        """ initializes rnn module """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.hidden = None

    def forward(self, x, return_activations=False):
        """ forward pass through network """

        x = x.long()

        embedded = self.embedding(x)

        embedded = embedded.to(torch.float32)

        if self.hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        else:
            h0 = self.hidden

        output, h_n = self.rnn(embedded, h0)
        
        self.hidden = h_n # shape of h_n is (num_layers, batch_size, hidden_dim)


        out = self.output(output)

        if return_activations:
            return out, output, h_n

        else:
            return out
    
    def reset_hidden(self):
        """Reset the hidden state"""
        self.hidden = None
    
    
    def init_hidden(self, batch_size, device=torch.device('cpu')):
        """ initializes hidden state """
        # For RNN, only need one hidden state tensor, not a tuple
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        

    def detach_hidden(self):
        """ detaches hidden states """
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())    

    