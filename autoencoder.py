import torch.nn as nn
import torch

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default precision based on device
        if self.device.type == "cuda":
            self.dtype = torch.float16  # Use half precision on GPU by default
        else:
            self.dtype = torch.float32
            
        print(f"Using device: {self.device}, dtype: {self.dtype}")

        self.encoder_layer = nn.Linear(stream_width, hidden_width)
        self.decoder_layer = nn.Linear(hidden_width, stream_width)
        self.relu = nn.ReLU()

        # initializing using Kaiming Unifform init. 
        nn.init.kaiming_uniform_(self.encoder_layer.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.decoder_layer.weight, nonlinearity="linear")

        self.to(self.device)
        self.to(self.dtype)
        
    def forward(self, x):
        """ forward pass function for the autoencoder """

        x = x.to(self.device, self.dtype)

        # hidden higher dimensional representation
        representation = self.relu(self.encoder_layer(x))

        # reconstruction based on representation
        reconstruction = self.decoder_layer(representation)

        return reconstruction
    


        


