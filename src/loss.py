import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import Autoencoder, CNN_Autoencoder, Encoder, Decoder

class TensorisedAEloss(nn.Module):
    def __init__(self, in_feature, embed_l, reg, num_clusters=2, linear=True, CNN=False):
        super(TensorisedAEloss, self).__init__()
        self.AE = nn.ModuleList()
        # add num_clusters AE
        self.n_clust = num_clusters
        for i in range(num_clusters):
            # note: this should be written better so that the network is passed
            if CNN:
                self.AE.append(CNN_Autoencoder(embed_l[0]))
            else:
                self.AE.append(Autoencoder(in_feature, embed_l, linear))

        self.mse = nn.MSELoss()
        self.reg = reg
        self.CNN = CNN

    def forward(self, X, centers, i, clust_assign, X_out=None):
        if X_out == None:
            X_out = X

        loss = 0
        loss_clust_idx = -1
        loss_clust = torch.inf

        for j in range(self.n_clust):
            x = X - centers[j]
            x_out = X_out - centers[j]

            if self.CNN:
                # this can def be optimized
                x = x.reshape(28, 28)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)

                x_out = x_out.reshape(28, 28)
                x_out = torch.unsqueeze(x_out, dim=0)
                x_out = torch.unsqueeze(x_out, dim=0)

                l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].enc2(self.AE[j].enc1(x)))))
            else:
                l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].get_embedding(x))))
    
            if loss_clust > l:
                loss_clust = l
                loss_clust_idx = j
            l = clust_assign[j][i] * l
            loss += l
        return loss, loss_clust_idx
    
    
class PartSharedLoss(nn.Module):
    def __init__(self, in_feature, embed_l, reg, num_clusters=2, linear=True, CNN=False, sig=False):
        super(PartSharedLoss, self).__init__()
        self.n_clust = num_clusters
        self.CNN = CNN
        num_layers = len(embed_l)

        # Shared part
        if CNN:
            self.shared_encoder = CNN_Encoder(fst_embed)
            self.shared_decoder = CNN_Decoder(fst_embed)
        else: 
            self.shared_encoder = nn.ModuleList()
            for l in range(num_layers-1):
                if l == 0:
                    self.shared_encoder.append(Encoder(in_feature, embed_l[l], linear))
                else:
                    self.shared_encoder.append(Encoder(embed_l[l-1], embed_l[l], linear))
            self.shared_decoder = nn.ModuleList() 
            for l in range(len(embed_l)-2,-1,-1):
                if l == 0:
                    self.shared_decoder.append(Decoder(embed_l[l], in_feature, sig=sig))
                else: 
                    self.shared_decoder.append(Decoder(embed_l[l], embed_l[l-1], sig=sig))

        # Tensorized part
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_clusters):
            self.encoders.append(Encoder(embed_l[num_layers-2], embed_l[num_layers-1], linear))
            self.decoders.append(Decoder(embed_l[num_layers-1], embed_l[num_layers-2], sig))

        self.mse = nn.MSELoss()
        self.reg = reg
        
    def forward(self, X, centers, i, clust_assign, X_out=None):
        if X_out == None:
            X_out = X

        loss = 0
        loss_clust_idx = -1
        loss_clust = torch.inf

        for j in range(self.n_clust):
            x = X - centers[j]
            x_out = X_out - centers[j]
             
            if self.CNN:
                # this can def be optimized
                x = x.reshape(28, 28)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)

                x_out = x_out.reshape(28, 28)
                x_out = torch.unsqueeze(x_out, dim=0)
                x_out = torch.unsqueeze(x_out, dim=0)
 
            for enc_layer in self.shared_encoder:
                x = enc_layer(x)
            encoded1 = x
            encoded2 = self.encoders[j](encoded1)
            reconstructed1 = self.decoders[j](encoded2)
            reconstructed2 = reconstructed1
            for dec_layer in self.shared_decoder:
                reconstructed2 = dec_layer(reconstructed2)
            
            l = self.mse(reconstructed2, x_out) + (self.reg * torch.square(torch.norm(encoded2)))

            if loss_clust > l:
                loss_clust = l
                loss_clust_idx = j
            l = clust_assign[j][i] * l

            loss += l
        return loss, loss_clust_idx
    
def vae_loss(x, x_hat, mean, log_var, reg=0.5, mse=False):
    if mse:
        reproduction_loss = nn.functional.mse_loss(x_hat, x)
    else:
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - reg * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def tensorised_vae_loss(model, X, centers, clust_assign, n_clusters=3, reg=0.5, linear_output=False, mse=False, cnn_vae=False):
    loss = 0
    loss_clust_idx = -1
    loss_clust = torch.inf

    for j in range(n_clusters):
        x = X - centers[j].view(*X.shape)
        if not linear_output:
            x = nn.functional.sigmoid(x)
        if cnn_vae:
            x = x.view(1,*x.shape)
        x_hat, mean, log_var = model(x, j)
        l = vae_loss(x, x_hat, mean, log_var, reg=reg, mse=mse)
        
        if loss_clust > l:
            loss_clust = l
            loss_clust_idx = j
#         l = clust_assign[j][i] * l
#         loss += l
    loss += loss_clust
    return loss, loss_clust_idx

def tensorised_rbm_loss(rbm, X, centers, clust_assign, n_clusters=3, n_vis=784):
    loss = 0
    loss_clust_idx = -1
    loss_clust = torch.inf

    for j in range(n_clusters):
        x = X - centers[j].view(*X.shape)
        data = torch.autograd.Variable(x.view(-1,n_vis))
        sample_data = x.view(-1,n_vis) #data.bernoulli()
        
        v,v1,h1 = rbm(sample_data, j)
        l = rbm.free_energy(v,j) - rbm.free_energy(v1,j)
        if loss_clust > l:
            loss_clust = l
            loss_clust_idx = j
    loss += loss_clust
    return loss, loss_clust_idx