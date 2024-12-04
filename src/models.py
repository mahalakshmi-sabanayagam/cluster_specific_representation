import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Autoencoder(nn.Module):
    def __init__(self, in_feature, embed_l, linear=True):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc = nn.ModuleList()
        for l in range(len(embed_l)):
            if l == 0:
                self.enc.append(nn.Linear(in_features=in_feature, out_features=embed_l[l], bias=False))
            else: 
                self.enc.append(nn.Linear(in_features=embed_l[l-1], out_features=embed_l[l], bias=False))
        # decoder
        self.dec = nn.ModuleList()
        for l in range(len(embed_l)-1,-1,-1):
            if l == 0:
                self.dec.append(nn.Linear(in_features=embed_l[l], out_features=in_feature, bias=False))
            else: 
                self.dec.append(nn.Linear(in_features=embed_l[l], out_features=embed_l[l-1], bias=False))
        self.linear = linear

    def forward(self, x):
        for i,enc_layer in enumerate(self.enc):
            x = enc_layer(x)
            if self.linear == False:
                x = F.relu(x)
        for i,dec_layer in enumerate(self.dec):
            x = dec_layer(x)
        return x
    
    def get_embedding(self, x):
        for i,enc_layer in enumerate(self.enc):
            x = enc_layer(x)
            if self.linear == False:
                x = F.relu(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(in_features=in_feature, out_features=embed, bias=False)
        self.linear = linear
#         nn.init.orthogonal_(self.enc1.weight)

    def forward(self, x):
        x = self.enc1(x)
        if not self.linear:
            x = F.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed, out_feature, sig=False):
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=embed, out_features=out_feature, bias=False)
        self.sig=sig

    def forward(self, x):
        x = self.dec1(x)
        if self.sig:
            x = F.sigmoid(x)
        return x
    
class CNN_Encoder(nn.Module):
    def __init__(self, embed):
        super(CNN_Encoder,self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, embed),
            # 10
            # nn.Softmax(),
            )
    def forward(self, x):
            enc = self.encoder(x)
            return enc
    
class CNN_Decoder(nn.Module):
    def __init__(self, embed):
        super(CNN_Decoder,self).__init__()
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(embed, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
            dec = self.decoder(x)
            return dec
 
class CNN_Autoencoder(nn.Module):
    def __init__(self,embed):
        super(CNN_Autoencoder,self).__init__()
        self.enc1 = CNN_Encoder(embed)
        self.dec1 = CNN_Decoder(embed)
    
    def forward(self, x):
        enc = self.enc1(x)
        dec = self.dec1(enc)
        return dec
    
class VAE(nn.Module):
    # code adapted from https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, embedding_dim=2, linear_output=False, tensorization=None, device="cpu"):
        super(VAE, self).__init__()
        self.linear_output = linear_output
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.tensorization = tensorization
        if tensorization!=None:
            self.mean_layer = nn.ModuleList()
            self.logvar_layer = nn.ModuleList()
            self.latent_layer = nn.ModuleList()
            for i in range(self.tensorization):
                self.mean_layer.append(nn.Linear(latent_dim,embedding_dim))
                self.logvar_layer.append(nn.Linear(latent_dim, embedding_dim))
                self.latent_layer.append(nn.Linear(embedding_dim, latent_dim))
        else:
            self.mean_layer = nn.Linear(latent_dim, embedding_dim)
            self.logvar_layer = nn.Linear(latent_dim, embedding_dim)
        
        # decoder
        if tensorization!=None:
            self.decoder = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, input_dim),
                )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(2, latent_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, input_dim),
                )
     
    def encode(self, x, clust_idx=None):
        x = self.encoder(x)
        if clust_idx != None:
            mean, logvar = self.mean_layer[clust_idx](x), self.logvar_layer[clust_idx](x)
        else:
            mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x, clust_idx=None):
        if clust_idx != None:
            x = self.latent_layer[clust_idx](x)
        if self.linear_output:
            return self.decoder(x)
        else:
            return F.sigmoid(self.decoder(x))

    def forward(self, x, clust_idx=None):
        mean, log_var = self.encode(x, clust_idx)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z, clust_idx)
        return x_hat, mean, log_var

class VAE_CNN(nn.Module):
    # code adapted from https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/ 
    def __init__(self, image_size=32, embedding_dim=2, linear_output=False, tensorization=None, device="cpu"):
        super(VAE_CNN, self).__init__()
        self.linear_output = linear_output
        self.tensorization = tensorization
        self.device = device
        self.latent_space = (128, image_size // 8, image_size // 8)
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
            )
        
        latent_dim = self.latent_space[0] * self.latent_space[1] * self.latent_space[2]
        if tensorization!=None:
            self.mean_layer = nn.ModuleList()
            self.logvar_layer = nn.ModuleList()
            self.decoder_fc = nn.ModuleList()
            for i in range(self.tensorization):
                self.mean_layer.append(nn.Linear(latent_dim, embedding_dim))
                self.logvar_layer.append(nn.Linear(latent_dim, embedding_dim))
                self.decoder_fc.append(nn.Linear(embedding_dim, latent_dim))
        else:
            self.mean_layer = nn.Linear(latent_dim, embedding_dim)
            self.logvar_layer = nn.Linear(latent_dim, embedding_dim)
            self.decoder_fc = nn.Linear(embedding_dim, latent_dim)

        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            )
        
    def encode(self, x, clust_idx=None):
        x = self.encoder(x)
        if clust_idx != None:
            mean, logvar = self.mean_layer[clust_idx](x), self.logvar_layer[clust_idx](x)
        else:
            mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x, clust_idx=None):
        if self.linear_output:
            return self.decoder(x)
        else:
            return F.sigmoid(self.decoder(x))

    def forward(self, x, clust_idx=None):
        mean, log_var = self.encode(x, clust_idx)
        z = self.reparameterization(mean, log_var)
        if clust_idx != None:
            x = self.decoder_fc[clust_idx](z)
        else:
            x = self.decoder_fc(z)
        x = x.view(-1, *self.latent_space)
        x_hat = self.decode(x, clust_idx)
        return x_hat, mean, log_var
        
class RBM(nn.Module):
   #code adapted from https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
    def __init__(self, n_vis=784, n_hin=500, k=5, tensorization=None, device="cpu"):
        super(RBM, self).__init__()
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.k = k
        self.device = device
        if tensorization!=None:
            self.W = nn.ParameterList()
            self.h_bias = nn.ParameterList()
            for i in range(tensorization):
                self.W.append(nn.Parameter(torch.randn(n_hin,n_vis)*1e-2))
                self.h_bias.append(nn.Parameter(torch.zeros(n_hin)))
        else:
            self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
            self.h_bias = nn.Parameter(torch.zeros(n_hin))
    
    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size())).to(self.device)))
    
    def v_to_h(self, v, idx=None):
        if idx!=None:
            p_h = F.sigmoid(F.linear(v,self.W[idx],self.h_bias[idx]))
        else:   
            p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
    def h_to_v(self, h ,idx=None):
        if idx!=None:
            p_v = F.sigmoid(F.linear(h,self.W[idx].t(),self.v_bias))
        else:    
            p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
    def forward(self, v, idx=None):
        pre_h1,h1 = self.v_to_h(v, idx)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_, idx)
            pre_h_,h_ = self.v_to_h(v_, idx)
        
        return v,v_,h_
    
    def free_energy(self, v, idx=None):
        if idx!=None:
            vbias_term = v.mv(self.v_bias)
            wx_b = F.linear(v,self.W[idx],self.h_bias[idx])
            hidden_term = wx_b.exp().add(1).log().sum(1)
        else:
            vbias_term = v.mv(self.v_bias)
            wx_b = F.linear(v,self.W,self.h_bias)
            hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

