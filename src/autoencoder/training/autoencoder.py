import torch.nn as nn

class SparseAutoEncoder(nn.Module):
    """ class representing the sparse autoencoder that will learn activation patterns """

    def __init__(self, stream_width, hidden_width):
        """ initializes autoencoder
        Params
        ------
        stream_width : int
            int representing parameters of input stream

        hidden_width : int
            int representing number of hidden params - must be GREATER 
            than stream_width
           """

        super().__init__()

        self.encoder_layer = nn.Linear(stream_width, hidden_width)
        self.decoder_layer = nn.Linear(hidden_width, stream_width)
        self.relu = nn.ReLU()

        # initializing using Kaiming Unifform init. 
        nn.init.kaiming_uniform_(self.encoder_layer.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.decoder_layer.weight, nonlinearity="linear")
        
    def forward(self, x):
        """ forward pass function for the autoencoder """

        # hidden higher dimensional representation
        representation = self.relu(self.encoder_layer(x))

        # reconstruction based on representation
        reconstruction = self.decoder_layer(representation)

        return reconstruction