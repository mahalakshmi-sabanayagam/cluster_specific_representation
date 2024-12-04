import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.cluster import adjusted_rand_score
from src.loss import TensorisedAEloss, PartSharedLoss, tensorised_vae_loss, vae_loss, tensorised_rbm_loss
from src.models import VAE, VAE_CNN, RBM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_AE(net, X, Y, lr=0.1, epochs=100, CNN=False, X_out=None):
    X_out = X_out if X_out is not None else X
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.1)
    X, Y, X_out = X.to(device), Y.to(device), X_out.to(device)
    train_loss = []

    print("Number of parameters in AE: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(X.shape[0]):
            x, x_out = X[i], X_out[i]
            if CNN:
                x = x.reshape(28, 28).unsqueeze(0).unsqueeze(0)
                x_out = x_out.reshape(28, 28).unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            out = net(x.float())
            loss = criterion(out, x_out.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / X.shape[0]
        train_loss.append(epoch_loss)

    return train_loss


def train_TAE(X, Y, n_clusters=2, lr=0.1, reg=0, embed_l=[2], epochs=100, number_of_batches=1, linear=True, CNN=False, shared=False, X_out=None, printing=False, sig=False):
    X_out = X_out if X_out is not None else X
    X, Y, X_out = X.to(device), Y.to(device), X_out.to(device)
    train_loss = []

    centers, indices = kmeans_plusplus(X_out.cpu().detach().numpy(), n_clusters=n_clusters, random_state=20)
    clust = torch.cat([torch.norm(X_out - torch.tensor(center).to(device), dim=1).reshape(-1, 1) for center in centers], dim=1)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, X.shape[0]], dtype=torch.float64).to(device)
    for i in range(X.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ X_out.float()
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float).to(device)
    centers = centers / norm

    if shared:
        net = PartSharedLoss(X.shape[1], embed_l=embed_l, reg=reg, num_clusters=n_clusters, linear=linear, CNN=CNN, sig=sig).to(device)
    else: 
        net = TensorisedAEloss(X.shape[1], embed_l=embed_l, reg=reg, num_clusters=n_clusters, linear=linear, CNN=CNN).to(device)
        
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    print("Number of parameters in TAE: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    for epoch in range(epochs):
        total_loss = 0
        batch_size = int(X.shape[0] / number_of_batches)
        
        for b in range(int(X.shape[0] / batch_size)):
            optimizer.zero_grad()
            temp_idx = []
            batch_loss = 0

            for i in range(batch_size):
                j = b * batch_size + i
                loss_sample, idx = net(X[j].float(), centers, j, clust_assign, X_out[j].float())
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample.item()

            batch_loss /= batch_size
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for k in range(batch_size):
                kb = b * batch_size + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1

            new_centers = clust_assign.float() @ X_out.float()
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float).to(device)
            new_centers /= new_norm
            centers = (b * centers + new_centers) / (b + 1)

        epoch_loss = total_loss / X.shape[0]
        if printing:
            print('epoch', epoch, 'loss', epoch_loss)
            print("ARI:", adjusted_rand_score(torch.argmax(clust_assign, axis=0).cpu().detach().numpy(), Y.cpu().detach().numpy()))
        elif (epoch + 1) == epochs:
            print('epoch', epoch, 'loss', epoch_loss)
        train_loss.append(epoch_loss)

    return net, train_loss, clust_assign, X, Y, X_out


def train_CTAE(X, Y, n_clusters=2, lr=0.1, reg=0, embed=2, epochs=100, number_of_batches=1, linear=True, CNN=False, X_out=None):
    X_out = X_out if X_out is not None else X
    X, Y, X_out = X.to(device), Y.to(device), X_out.to(device)
    train_loss = []

    centers, indices = kmeans_plusplus(X_out.cpu().detach().numpy(), n_clusters=n_clusters, random_state=20)
    clust = torch.cat([torch.norm(X_out - torch.tensor(center).to(device), dim=1).reshape(-1, 1) for center in centers], dim=1)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, X.shape[0]], dtype=torch.float64).to(device)
    for i in range(X_out.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ X_out.float()
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X_out.shape[1], dtype=torch.float).to(device)
    centers /= norm

    net = TensorisedCAEloss(X_out.shape[1], embed, num_clusters=n_clusters).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        batch_size = int(X.shape[0] / number_of_batches)
        
        for b in range(number_of_batches):
            optimizer.zero_grad()
            temp_idx = []
            batch_loss = 0

            for i in range(batch_size):
                j = b * batch_size + i
                loss_sample, idx = net(X[j].float(), centers, j, clust_assign, X_out[j].float())
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample.item()

            batch_loss /= batch_size
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for k in range(batch_size):
                kb = b * batch_size + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1

            new_centers = clust_assign.float() @ X_out.float()
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float).to(device)
            new_centers /= new_norm
            centers = (b * centers + new_centers) / (b + 1)

        epoch_loss = total_loss / X.shape[0]
        print('epoch', epoch, 'loss', epoch_loss)
        train_loss.append(epoch_loss)

    return net, train_loss, clust_assign, X, Y, X_out

def train_tensorVAE(train_loader, lr=1e-3, epochs=200, n_clusters=3, batch_size=1, reg=0.5, linear_output=False, mse=False, print_every=10, cnn_vae=False, device="cpu"):
    train_dataset_array = np.array([train_loader.dataset[i][0].numpy().reshape(-1) for i in range(len(train_loader.dataset))])
    centers, indices = kmeans_plusplus(train_dataset_array, n_clusters=n_clusters, random_state=20)
    clust = torch.cat([torch.norm(torch.tensor(train_dataset_array).to(device) - torch.tensor(center).to(device), dim=1).reshape(-1, 1) for center in centers], dim=1)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, train_dataset_array.shape[0]], dtype=torch.float64).to(device)
    for i in range(train_dataset_array.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ torch.tensor(train_dataset_array).to(device)
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, train_dataset_array.shape[1], dtype=torch.float).to(device)
    centers = centers / norm
       
    if cnn_vae:
        model = VAE_CNN(tensorization=n_clusters, linear_output=linear_output, device=device).to(device)
    else:
        model = VAE(tensorization=n_clusters, linear_output=linear_output, device=device).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for b, (x, y) in enumerate(train_loader): #for b in range(int(X.shape[0] / batch_size)):
            if not cnn_vae:
                x = x.reshape(batch_size, -1)
            x = x.to(device)
            
            optimizer.zero_grad()
            batch_loss = 0
            temp_idx = []
            for i in range(batch_size):
                loss_sample, idx = tensorised_vae_loss(model, x[i], centers, clust_assign, n_clusters=n_clusters, reg=reg, 
                                                       linear_output=linear_output, mse=mse, cnn_vae=cnn_vae)
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample.item()

            batch_loss /= batch_size
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for k in range(batch_size):
                kb = b * batch_size + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1

            new_centers = clust_assign.float() @ torch.tensor(train_dataset_array).to(device)
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, train_dataset_array.shape[1], dtype=torch.float).to(device)
            new_centers /= (new_norm+1e-8)
            centers = (b * centers + new_centers) / (b + 1)

        epoch_loss = total_loss / train_dataset_array.shape[0]
        if epoch%print_every == 0:
            print('epoch', epoch, 'loss', epoch_loss)
        train_loss.append(epoch_loss)

    return model, train_loss, clust_assign

def train_VAE(train_loader, lr=1e-3, epochs=200, batch_size=1, reg=0.5, linear_output=False, mse=False, print_every=50, cnn_vae=False, device="cpu"):
    if cnn_vae:
        model = VAE_CNN(linear_output=linear_output, device=device).to(device)
    else:
        model = VAE(linear_output=linear_output, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if not cnn_vae:
                x = x.reshape(batch_size, -1)
            x = x.to(device) #x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = vae_loss(x, x_hat, mean, log_var, reg=reg, mse=mse)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        if epoch%print_every == 0:
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return model, overall_loss

def train_tensorRBM(train_loader, n_vis=784, n_hin=500, k=1, n_clusters=3, epochs=10, lr=0.1, device="cpu"):
    train_dataset_array = np.array([train_loader.dataset[i][0].numpy().reshape(-1) for i in range(len(train_loader.dataset))])
    centers, indices = kmeans_plusplus(train_dataset_array, n_clusters=n_clusters, random_state=20)
    clust = torch.cat([torch.norm(torch.tensor(train_dataset_array).to(device) - torch.tensor(center).to(device), dim=1).reshape(-1, 1) for center in centers], dim=1)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, train_dataset_array.shape[0]], dtype=torch.float64).to(device)
    for i in range(train_dataset_array.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ torch.tensor(train_dataset_array).to(device)
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, train_dataset_array.shape[1], dtype=torch.float).to(device)
    centers = centers / norm
    
    rbm = RBM(k=k, tensorization=n_clusters).to(device)
#     optimizer = torch.optim.SGD(rbm.parameters(),0.1)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=lr)
    train_loss = []
    for epoch in range(epochs):
        total_loss = 0
        samples_updated = 0
        for b, (x,y) in enumerate(train_loader):
            x = x.to(device)
            cur_batch_size = x.shape[0]
            optimizer.zero_grad()
            batch_loss = 0
            temp_idx = []
            for i in range(cur_batch_size):
                loss_sample, idx = tensorised_rbm_loss(rbm, x[i], centers, clust_assign, n_clusters=n_clusters, n_vis=n_vis)
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample#.item()
            batch_loss /= len(train_loader)
            batch_loss.backward(retain_graph=True)
            optimizer.step()
            
            for k in range(cur_batch_size):
                kb = samples_updated + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1
            samples_updated += cur_batch_size
                
            new_centers = clust_assign.float() @ torch.tensor(train_dataset_array).to(device)
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, train_dataset_array.shape[1], dtype=torch.float).to(device)
            new_centers /= (new_norm+1e-8)
            centers = (b * centers + new_centers) / (b + 1)

        epoch_loss = total_loss / train_dataset_array.shape[0]
        print('epoch', epoch, 'loss', epoch_loss)
        train_loss.append(epoch_loss)

    return rbm, train_loss, clust_assign

def train_RBM(train_loader, n_vis=784, n_hin=500, k=1, epochs=10, lr=0.1, device="cpu"):
    rbm = RBM(n_vis=n_vis, n_hin=n_hin, k=k, device=device).to(device)
#     optimizer = torch.optim.SGD(rbm.parameters(),lr=lr)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=lr)
    for epoch in range(epochs):
        loss_ = 0
        for _, (data,target) in enumerate(train_loader):
            batch_size = data.shape[0]
            data = torch.autograd.Variable(data.view(-1,n_vis))
            sample_data = data.bernoulli().to(device)

            v,v1,h1 = rbm(sample_data)
            loss = rbm.free_energy(v) - rbm.free_energy(v1)
            loss_+= loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", loss_/batch_size)
    return rbm, loss_/batch_size