import numpy as np
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.loss import vae_loss
from src.models import VAE, VAE_CNN
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import make_grid
from sklearn.metrics.cluster import adjusted_rand_score


FIGURE_DIR = Path('results/figures/')

def best_hyperparam_clust(csv_file, models=['kmeans', 'ae1', 'ae2', 'ae3', 'tae1', 'tae2', 'ptae']):
    '''
    assumes csv_file with
    fields = ["data", "rep", "X", "Y", "shape", "noise",
             "kmeans_labels", "kmeans_ari",
             "ae1_num_params", "ae1_lr", "ae1_epochs", "ae1_final_train_loss", "ae1_labels", "ae1_ari", 
             "ae2_num_params", "ae2_lr", "ae2_epochs", "ae2_final_train_loss", "ae2_labels", "ae2_ari", 
             "ae3_num_params", "ae3_lr", "ae3_epochs", "ae3_final_train_loss", "ae3_labels", "ae3_ari",
             "tae1_num_params", "tae1_lr", "tae1_epochs", "tae1_final_train_loss", "tae1_labels", "tae1_ari",
             "tae2_num_params", "tae2_lr", "tae2_epochs", "tae2_final_train_loss", "tae2_labels", "tae2_ari",
             "ptae_num_params", "ptae_lr", "ptae_epochs", "ptae_final_train_loss", "ptae_labels", "ptae_ari", 
             ]
    finds the best lr for each model 
    res_df : returns a dataframe with the best lr for each model and data
    '''
    res_df = pd.read_csv(csv_file)
    df = res_df.drop(columns=['X', 'Y', 'shape', 'noise', 
                  'kmeans_labels', 'ae1_labels', 'ae2_labels', 'ae3_labels',
                  'tae1_labels', 'tae2_labels', 'ptae_labels',])
    # find the best lr for each dataset, epochs=150 for all the settings in synthetic and 200 for real
    # create a new df with data, model, lr, ari_mean, ari_std, final_train_loss_mean, final_train_loss_std, num_params
    res_data_l = []
    res_model_l = []
    res_lr_l = []
    res_ari_mean = []
    res_ari_std = []
    res_final_train_loss_mean = []
    res_final_train_loss_std = []
    res_num_params = []

    data = list(set(df['data']))
    for m in models:
        if m=='kmeans':
            cols = ['data', 'kmeans_ari']
        else:
            cols = ['data', m+'_lr', m+'_final_train_loss', m+'_ari', m+'_num_params']
        gdf = df[cols]
        if m=='kmeans':
            gdf = gdf.groupby(by=['data']).agg({
                                            m+'_ari': ['mean', 'sem'],
                                            }).reset_index()
        else:
            gdf = gdf.groupby(by=['data', m+'_lr']).agg({
                                            m+'_final_train_loss': ['mean', 'sem'],
                                            m+'_ari': ['mean', 'sem'],
                                            m+'_num_params': ['mean'],
                                            }).reset_index()
        gdf.columns = ['_'.join(col).strip() for col in gdf.columns.values]
        for d in data:
            bdf = gdf[gdf['data_'] == d]
            best_df = bdf[bdf[m+'_ari_mean']== bdf[m+'_ari_mean'].max() ]
            res_data_l.append(d)
            res_model_l.append(m)
            if m!='kmeans':
                res_lr_l.append(best_df[m+'_lr_'].values.tolist())
                res_final_train_loss_mean.append(best_df[m+'_final_train_loss_mean'].values.tolist()[0])
                res_final_train_loss_std.append(best_df[m+'_final_train_loss_sem'].values.tolist()[0])
                res_num_params.append(best_df[m+'_num_params_mean'].values.tolist()[0])
            else:
                res_lr_l.append('-')
                res_final_train_loss_mean.append('-')
                res_final_train_loss_std.append('-')
                res_num_params.append(0)
            res_ari_mean.append(best_df[m+'_ari_mean'].values.tolist()[0])
            res_ari_std.append(best_df[m+'_ari_sem'].values.tolist()[0])
    res_df = pd.DataFrame({'data': res_data_l,
                         'model': res_model_l, 
                         'lr': res_lr_l,
                         'final_train_loss_mean': res_final_train_loss_mean,
                         'final_train_loss_sem': res_final_train_loss_std,
                         'ari_mean': res_ari_mean,
                         'ari_sem': res_ari_std,
                         'num_params': res_num_params
                        })
    return res_df

def best_hyperparam_denoise(csv_file, models=['ae1', 'ae2', 'ae3', 'tae1', 'tae2', 'ptae']):
    '''
    assumes csv_file with
    fields = ["data", "rep", "X", "Y", "shape", "noise",
         "ae1_num_params", "ae1_lr", "ae1_epochs", "ae1_final_train_loss", #"ae1_labels", "ae1_ari", 
         "ae2_num_params", "ae2_lr", "ae2_epochs", "ae2_final_train_loss", #"ae2_labels", "ae2_ari", 
         "ae3_num_params", "ae3_lr", "ae3_epochs", "ae3_final_train_loss", #"ae3_labels", "ae3_ari",
         "tae1_num_params", "tae1_lr", "tae1_epochs", "tae1_final_train_loss", #"tae1_labels", "tae1_ari",
         "tae2_num_params", "tae2_lr", "tae2_epochs", "tae2_final_train_loss", #"tae2_labels", "tae2_ari",
         "ptae_num_params", "ptae_lr", "ptae_epochs", "ptae_final_train_loss", #"ptae_labels", "ptae_ari", 
         ]
    finds the best lr for each model 
    res_df : returns a dataframe with the best lr for each model and data
    '''
    res_df = pd.read_csv(csv_file)
    df = res_df.drop(columns=['X', 'Y', 'shape', 'noise'])
    # find the best lr for each dataset, epochs=150 for all the settings in synthetic and 200 for real
    # create a new df with data, model, lr, ari_mean, ari_std, final_train_loss_mean, final_train_loss_std, num_params
    res_data_l = []
    res_model_l = []
    res_lr_l = []
    res_final_train_loss_mean = []
    res_final_train_loss_std = []
    res_num_params = []

    data = list(set(df['data']))
    for m in models:
        cols = ['data', m+'_lr', m+'_final_train_loss',  m+'_num_params']
        gdf = df[cols]
        gdf = gdf.groupby(by=['data', m+'_lr']).agg({
                                            m+'_final_train_loss': ['mean', 'sem'],
                                            m+'_num_params': ['mean'],
                                            }).reset_index()
        gdf.columns = ['_'.join(col).strip() for col in gdf.columns.values]
        for d in data:
            bdf = gdf[gdf['data_'] == d]
            best_df = bdf[bdf[m+'_final_train_loss_mean']== bdf[m+'_final_train_loss_mean'].min() ]
            res_data_l.append(d)
            res_model_l.append(m)
            res_lr_l.append(best_df[m+'_lr_'].values.tolist())
            res_final_train_loss_mean.append(best_df[m+'_final_train_loss_mean'].values.tolist()[0])
            res_final_train_loss_std.append(best_df[m+'_final_train_loss_sem'].values.tolist()[0])
            res_num_params.append(best_df[m+'_num_params_mean'].values.tolist()[0])
    res_df = pd.DataFrame({'data': res_data_l,
                         'model': res_model_l, 
                         'lr': res_lr_l,
                         'final_train_loss_mean': res_final_train_loss_mean,
                         'final_train_loss_sem': res_final_train_loss_std,
                         'num_params': res_num_params
                        })
    return res_df 

def plot_ae_bar(df, y_col = "ari", mean=True, sem=True, ylabel=None, yscale_log=False, legend_loc="upper left",  save_file=None):
    '''
    models is asusmed to be ['kmeans', 'ae1', 'ae2', 'ae3', 'tae1', 'tae2', 'ptae']
    y_col = "ari" or "num_params" or "final_train_loss"
    '''
    n_palette_kmeans = [sns.color_palette("colorblind", 7)[1]]
    n_palette_ae = [sns.color_palette("mako",7)[i] for i in [4,5,6]]
    n_palette_tae = [sns.color_palette("rocket", 6)[i] for i in [4,5]]
    n_palette_ptae = [sns.color_palette("Set3", 7)[6]]
    models = list(set(df["model"]))
    if "kmeans" in models or "KMeans" in models:
        n_palette = n_palette_kmeans + n_palette_ae + n_palette_tae + n_palette_ptae
    else:
        n_palette = n_palette_ae + n_palette_tae + n_palette_ptae

    plot_y = y_col
    if mean:
        plot_y = y_col+'_mean'
    # Create the bar plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='data', y=plot_y, hue='model', data=df, palette=n_palette, saturation=1, alpha=0.8)
    ax.legend_.set_title(None)
    plt.xlabel('')
    if ylabel:
        plt.ylabel(ylabel)
    if yscale_log:
        plt.yscale("log")
    plt.legend(loc=legend_loc)
          
    # Get the total number of bars in a single group
    n_bars = len(df['model'].unique())
    # The width of a group of bars
    bar_width = 0.8
    # Spacing between each group
    group_width = n_bars * bar_width
    # Individual bar width
    single_bar_width = bar_width / n_bars
    # Offset from the center of the group
    offset = (np.arange(n_bars) - np.arange(n_bars).mean()) * single_bar_width

    # Add error bars
    if sem:
        plot_err = y_col+'_sem'
        for i, dataset in enumerate(df['data'].unique()):
            for j, model in enumerate(df['model'].unique()):
                subset = df[(df['model'] == model) & (df['data'] == dataset)]
                # Calculate the position for each bar/errorbar
                # Note: +0.2 is added to align with the seaborn default, adjust this if necessary
                position = i - 0.2 + offset[j]+0.2
                plt.errorbar(x=position, y=subset[plot_y], yerr=subset[plot_err], fmt='none', c='black', capsize=5)
    if save_file:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_DIR/save_file, bbox_inches='tight')
    plt.show()
        
def generate_digit_vae(model, mean, var, digit_size=28, clust_idx=None, cnn_vae=False, device="cpu"):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    if cnn_vae:
        if clust_idx!=None:
            z = model.decoder_fc[clust_idx](z_sample)
        else:
            z = model.decoder_fc(z_sample)
        z = z.view(-1, *model.latent_space)
        x_decoded = model.decode(z)
    else:
        x_decoded = model.decode(z_sample, clust_idx)
    digit = x_decoded.detach().cpu().reshape(digit_size, digit_size) # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()
    
def plot_latent_space_vae(model, idx=0, scale=1.0, n=10, digit_size=28, figsize=15, cnn_vae=False, device="cpu", save_file=None):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            if cnn_vae:
                if idx!=None:
                    z = model.decoder_fc[idx](z_sample)
                else:
                    z = model.decoder_fc(z_sample)
                z = z.view(-1, *model.latent_space)
                x_decoded = model.decode(z)
            else:
                x_decoded = model.decode(z_sample, idx)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    if idx == None:
        plt.title('VAE')
    else:
        plt.title('TVAE')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.axis('off')
    plt.imshow(figure, cmap="Greys_r")
    if save_file:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_DIR/save_file, bbox_inches='tight')
    plt.show()
    
def plot_reconstruction_vae(original, reconstruction, img_n=10, image_size=32, device="cpu"):
    # original : list of original images 
    # reconstruction : list of reconstructed images
    orig = original[:img_n]
    reconst = reconstruction[:img_n]
    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, img_n), axes_pad=0.1)

    for ax, im in zip(grid, orig+reconst):
        ax.imshow(im.reshape(image_size,image_size), cmap='gray')
        ax.axis('off')
    grid[0].set_ylabel("Original", rotation=0)
    grid[img_n].set_ylabel("Reconstruction", rotation=0)
    plt.show()
        
def get_reconstruction(model, loader, n_clusters=3, reg=0.1, mse=False, cnn_vae=False, device="cpu"):
    # 
    model.eval()
    reconstruction = []
    original = []
    true_label = []
    pred_label = []
    
    data = next(iter(loader))[0]
    data_label = next(iter(loader))[1]
    for i in range(len(data)):
        x = data[i]
        if cnn_vae:
            x = x.view(1, *x.shape).to(device)
        else:
            x = x.view(-1).to(device)
        loss = torch.inf
        rep = None
        pred = -1
        x_reconst = None
        if n_clusters==None:
            x_hat, mean, log_var = model(x)
            loss = vae_loss(x, x_hat, mean, log_var, reg=reg, mse=mse)
            rep = mean
            x_reconst = x_hat
        else:
            for j in range(n_clusters):
                x_hat, mean, log_var = model(x, j)
                l = vae_loss(x, x_hat, mean, log_var, reg=reg, mse=mse)
                if loss>l:
                    loss = l
                    rep = mean
                    pred = j
                    x_reconst = x_hat
            true_label.append(data_label[i].item())
            pred_label.append(pred)
        reconstruction.append(x_reconst.detach().cpu().numpy())
        original.append(x.detach().cpu().numpy())
    return original, reconstruction, true_label, pred_label

def plot_training_images(train_loader, num_samples = 25):
    # get 25 sample training images for visualization
    dataiter = iter(train_loader)
    image = next(dataiter)
    sample_images = [image[0][i,0] for i in range(num_samples)] 

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        ax.imshow(im, cmap='gray')
        ax.axis('off')

    plt.show()

def plot_vae_representation(model, loader, batch_size=64, n_clusters=3, device="cpu"):
    model.eval()
    representation = []
    true_label = []
    pred_label = []
    for batch_idx, (x, y) in enumerate(loader):
        x = x.reshape(batch_size, -1)
        x = x.to(device) #x.view(batch_size, x_dim).to(device)
        for i in range(batch_size):
            loss = torch.inf
            rep = None
            pred = -1
            for j in range(n_clusters):
                x_hat, mean, log_var = model(x[i], j)
                l = vae_loss(x[i], x_hat, mean, log_var)
                if loss>l:
                    loss = l
                    rep = mean
                    pred = j
            representation.append(rep.detach().cpu().tolist())
            true_label.append(y[i].item())
            pred_label.append(pred)
    true_label = np.array(true_label)
    true_label[true_label==9] = 2
    print("true label ", true_label)
    print("pred label ", pred_label)
    print("ARI ", adjusted_rand_score(true_label, np.array(pred_label)))
    representation = np.array(representation)
    representation = representation.reshape(-1, 2)
    true_label = np.array(true_label)
    true_label = true_label.reshape(-1)
    for g in np.unique(true_label):
        i = np.where(true_label == g)
        plt.scatter(representation[i,0],representation[i,1], label=g)
    plt.legend()

def plot_vae_representation_per_cluster(model, loader, cnn_vae=False, batch_size=64, clust_idx=0, figsize=3, device="cpu", save_file=None):
    model.eval()
    true_label = []
    j = clust_idx
    representation = []
    for batch_idx, (x, y) in enumerate(loader):
        if not cnn_vae:
            x = x.reshape(batch_size, -1)
        x = x.to(device) #x.view(batch_size, x_dim).to(device)
        for i in range(batch_size):
            if cnn_vae:
                X = x[i].view(1,*x[i].shape)
            else:
                X = x[i]
            x_hat, mean, log_var = model(X, j)
            representation.append(mean.detach().cpu().tolist())
            true_label.append(y[i].item())
    representation = np.array(representation)
    representation = representation.reshape(-1, 2)
    true_label = np.array(true_label)
    plt.figure(figsize=(figsize, figsize))
    if j != None:
        plt.scatter(representation[:,0],representation[:,1], c=sns.color_palette("pastel")[j])
    else:
        gt = np.unique(true_label)
        for g in range(len(gt)):
            i = np.where(true_label == gt[g])
            plt.scatter(representation[i,0],representation[i,1], label=gt[g], c=sns.color_palette("pastel")[g])
            plt.legend()
    m1, m2 = representation[:,0], representation[:,1]
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    from scipy import stats
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    plt.contour(X, Y, Z, alpha=0.3, colors='crimson')
    if save_file:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_DIR/save_file, bbox_inches='tight')
    plt.show()
    plt.close()


def load_vae_model(lr = 1e-3, epochs = 200, reg = 0.2, 
         mse = False, linear_output = False, n_clusters = 3,
        vae="tvae", cnn_vae=False, digit_size=32, device="cpu"):
    fname = f'./models/{vae}_lr{lr}_ep{epochs}_reg{reg}_mse{mse}_lout{linear_output}'
    if cnn_vae:
        model = VAE_CNN(tensorization=n_clusters, linear_output=linear_output, device=device).to(device)
    else:
        model = VAE(tensorization=n_clusters, linear_output=linear_output, device=device).to(device)
    model.load_state_dict(torch.load(fname, map_location=device))
    return model
    
def plot_rbm(rbm, loader, n_clusters=None, batch_size=64, figsize=3, savefile=0, fname="", device="cpu"):
    one_batch_x = next(iter(loader))[0]
    one_batch_y = next(iter(loader))[1]
    data_x = one_batch_x.view(-1,784).to(device)
    data_x = make_grid(data_x.view(batch_size,1,28,28).data)
    plt.figure(figsize=(figsize, figsize))
    plt.axis('off')
    npimg = np.transpose(data_x.cpu().detach().numpy(),(1,2,0))
    plt.imshow(npimg)
    plt.title('Original')
    if savefile:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_file = f"rbm_original_{fname}.pdf"
        plt.savefig(FIGURE_DIR/save_file, bbox_inches='tight')
    plt.show()
    plt.close()

    data = torch.autograd.Variable(one_batch_x.view(-1,784)).to(device)
    sample_data = data.bernoulli()
    if n_clusters == None:
        v,v1,h1 = rbm(sample_data)
        img = make_grid(v1.view(batch_size,1,28,28).data)
        npimg = np.transpose(img.cpu().detach().numpy(),(1,2,0))
        plt.figure(figsize=(figsize, figsize))
        plt.axis('off')
        plt.title('Reconstruction')
        plt.imshow(npimg) 
        if savefile:
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            save_file = f"rbm_reconst_{fname}.pdf"
            plt.savefig(FIGURE_DIR/save_file, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        for c in range(n_clusters):
            v,v1,h1 = rbm(sample_data,c)
            img = make_grid(v1.view(batch_size,1,28,28).data)
            npimg = np.transpose(img.cpu().detach().numpy(),(1,2,0))
            plt.figure(figsize=(figsize, figsize))
            plt.axis('off')
            plt.title(f'C{c+1} Reconstruction')
            plt.imshow(npimg)
            if savefile:
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                save_file = f"rbm_reconst_{fname}_{c}.pdf"
                plt.savefig(FIGURE_DIR/save_file, bbox_inches='tight')
            plt.show()
            plt.close()