import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 500),
            nn.Sigmoid(),
            nn.Linear(500, 250),
            nn.Sigmoid(),
            nn.Linear(250, 30),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.Sigmoid(),
            nn.Linear(250, 500),
            nn.Sigmoid(),
            nn.Linear(500, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out

def AE():
    return Autoencoder()

#net = AE()
#for name, parameters in net.named_parameters():
#    print(name)
