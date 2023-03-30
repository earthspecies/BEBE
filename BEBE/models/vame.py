# Adapted from https://github.com/LINCellularNeuroscience/VAME
# Under GPL 3.0 License

import yaml
import numpy as np
import pickle
import os
import shutil
from BEBE.models.model_superclass import BehaviorModel
from BEBE.models.preprocess import whitener_standalone
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import random
import tqdm


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class vame(BehaviorModel):
  def __init__(self, config):
    super(vame, self).__init__(config)
    print(f"Using {device} device")
        
    self.lr = self.model_config['learning_rate']
    self.n_epochs = 100
    self.downsizing_factor = self.get_downsizing_factor()
    self.temporal_window_samples = 2*int(self.model_config['time_window_sec'] * self.metadata['sr']) # Following the 2x from VAME
    self.seq_len_half = int(self.model_config['time_window_sec'] * self.metadata['sr'])
    self.batch_size = self.model_config['batch_size']
    self.n_clusters = self.config['num_clusters']
    self.beta = self.model_config['beta']
    self.zdims = self.model_config['zdims']
    self.prediction_decoder = self.model_config['prediction_decoder']
    self.prediction_steps = int(self.model_config['time_window_sec'] * self.metadata['sr']) # use time window = prediction window
    self.scheduler = self.model_config['scheduler']
    self.scheduler_step_size = self.model_config['scheduler_step_size']
    self.scheduler_gamma = self.model_config['scheduler_gamma']
    self.kmeans_loss = self.model_config['zdims'] # Uses all singular values
    self.kmeans_lambda = self.model_config['kmeans_lambda']
    self.seed = self.config['seed']
    
    torch.manual_seed(self.config['seed'])
    random.seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    
    self.whiten = self.model_config['whiten']
    self.whitener = whitener_standalone()
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters)
      
  def get_downsizing_factor(self):
    train_fps = self.config['train_data_fp']
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    data_len = np.shape(train_data)[0]    
    downsizing_factor = int((100 * data_len) / (self.model_config['n_train_steps'] * self.model_config['batch_size']))
    return downsizing_factor
    
  def fit(self):
    train_fps = self.config['train_data_fp']
    
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    
    if self.whiten:
      train_data = self.whitener.fit_transform(train_data)
      
    self.model = RNN_VAE(self.temporal_window_samples,self.zdims,np.shape(train_data)[1],self.prediction_decoder,self.prediction_steps, 256, 256, 256, 256, 0, 0, 0, False).to(device)
    
    self.model.train()

    
    train_dataset = SEQUENCE_DATASET(train_data, self.temporal_window_samples, True)
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
        
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.scheduler_gamma, patience=self.scheduler_step_size, threshold=1e-3, threshold_mode='rel', verbose=True)
    
    train_losses = []
    kl_losses = []
    km_losses = []
    mse_losses = []
    fut_losses = []
    learning_rates = []
    
    epochs = self.n_epochs
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        _, loss, km_loss, kl_loss, mse_loss, fut_loss = train_epoch(train_dataloader,
                                                                    t,
                                                                    self.model,
                                                                    optimizer,
                                                                    'linear',
                                                                    self.beta,
                                                                    2,
                                                                    4,
                                                                    self.temporal_window_samples,
                                                                    self.prediction_decoder,
                                                                    self.prediction_steps,
                                                                    scheduler,
                                                                    'mean', 
                                                                    'mean',
                                                                    self.kmeans_loss,
                                                                    self.kmeans_lambda,
                                                                    self.batch_size, 
                                                                    False, 
                                                                    downsizing_factor = self.downsizing_factor)
        
        train_losses.append(loss)
        kl_losses.append(kl_loss)
        km_losses.append(km_loss)
        mse_losses.append(mse_loss)
        fut_losses.append(fut_loss)
        scheduler.step(loss)
        
        # plot Loss
        plt.plot(train_losses, label = "total loss")
        plt.plot(kl_losses, label = "kl loss")
        plt.plot(km_losses, label = "kmeans loss")
        plt.plot(mse_losses, label = "mse loss")
        plt.plot(fut_losses, label = "fut loss")
        
        plt.legend()
        plt.title("Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
        plt.savefig(loss_fp)
        plt.close()
        
    
    print("Done with VAE optimization, fitting kmeans")
    train_dataloader = DataLoader(train_dataset, batch_size=8*self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    alpha = min(2.0 / (len(train_dataloader) + 1) , 1)
    max_no_improvement= 100
    converged = False
    ewa_inertia = None
    ewa_inertia_min = None
    no_improvement = 0
    inertias = []
    ewa_inertias = []
    
    for t in tqdm.tqdm(range(100)):
      if converged:
        break
      with torch.no_grad():
        for x in train_dataloader:
          x = x.permute(0,2,1)
          x = x[:,:self.seq_len_half,:].type('torch.FloatTensor').to(device)
          
          _, mu, _ = self.model.encode(x)
          mu = mu.cpu().numpy()
          self.kmeans.partial_fit(mu)
          
          # Compute an Exponentially Weighted Average of the inertia to
          # monitor the convergence while discarding minibatch-local stochastic
          # variability: https://en.wikipedia.org/wiki/Moving_average
          # https://github.com/scikit-learn/scikit-learn/blob/9aaed4987/sklearn/cluster/_kmeans.py#L1636
          if ewa_inertia is None:
              ewa_inertia = self.kmeans.inertia_
          else:
              ewa_inertia = ewa_inertia * (1 - alpha) + self.kmeans.inertia_ * alpha

          # Early stopping heuristic due to lack of improvement on smoothed
          # inertia
          if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
              no_improvement = 0
              ewa_inertia_min = ewa_inertia
          else:
              no_improvement += 1
              
          inertias.append(self.kmeans.inertia_)
          ewa_inertias.append(ewa_inertia)
          
          if len(ewa_inertias) % 100 == 0:
              plt.plot(inertias, label = "inertia")
              plt.plot(ewa_inertias, label = "ewa inertia")
              plt.legend()
              plt.title("Inertia across mini batches")
              loss_fp = os.path.join(self.config['output_dir'], 'inertia.png')
              plt.savefig(loss_fp)
              plt.close()

          if no_improvement >= max_no_improvement:
              print("Converged (lack of improvement in inertia)")
              converged = True
              break
              
    plt.plot(inertias, label = "inertia")
    plt.plot(ewa_inertias, label = "ewa inertia")
    plt.legend()
    plt.title("Inertia across mini batches")
    loss_fp = os.path.join(self.config['output_dir'], 'inertia.png')
    plt.savefig(loss_fp)
    plt.close()
            
    print("Done Training!")
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)

  def predict(self, data):
    if self.whiten:
      data = self.whitener.transform(data)

    dataset = SEQUENCE_DATASET(data, self.temporal_window_samples, False)
    dataloader = DataLoader(dataset, batch_size=8*self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    predictions = []

    self.model.eval()
    with torch.no_grad():
      for x in dataloader:
        x = x.permute(0,2,1)
        x = x[:,:self.seq_len_half,:].type('torch.FloatTensor').to(device)
        _, mu, _ = self.model.encode(x)
        mu = mu.cpu().numpy()
        clusters = self.kmeans.predict(mu)
        predictions.append(clusters)
    predictions = np.concatenate(predictions)
    print(np.shape(predictions))

    return predictions, None
    
    
      
  

#########
# Model
#########

class Encoder(nn.Module):
    def __init__(self, NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder):
        super(Encoder, self).__init__()
        
        self.input_size = NUM_FEATURES
        self.hidden_size = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.n_layers  = 2 
        self.dropout   = dropout_encoder
        self.bidirectional = True
        
        self.encoder_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)#UNRELEASED!
    
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers
        
    def forward(self, inputs):        
        outputs_1, hidden_1 = self.encoder_rnn(inputs)#UNRELEASED!
        
        hidden = torch.cat((hidden_1[0,...], hidden_1[1,...], hidden_1[2,...], hidden_1[3,...]),1)
        
        return hidden
    
    
class Lambda(nn.Module):
    def __init__(self,ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus):
        super(Lambda, self).__init__()
        
        self.hid_dim = hidden_size_layer_1*4
        self.latent_length = ZDIMS
        self.softplus = softplus
        
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)
        
        if self.softplus == True:
            print("Using a softplus activation to ensures that the variance is parameterized as non-negative and activated by a smooth function")
            self.softplus_fn = nn.Softplus()
        
    def forward(self, hidden):
        
        self.mean = self.hidden_to_mean(hidden)
        if self.softplus == True:
            self.logvar = self.softplus_fn(self.hidden_to_logvar(hidden))
        else:
            self.logvar = self.hidden_to_logvar(hidden)
        
        if self.training:
            std = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.mean), self.mean, self.logvar
        else:
            return self.mean, self.mean, self.logvar

      
class Decoder(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec):
        super(Decoder,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_rec
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_rec
        self.bidirectional = True
        
        self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW
        
        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor) # NEW
        self.hidden_to_output = nn.Linear(self.hidden_size*(2 if self.bidirectional else 1), self.num_features)
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0) # NEW
        
        hidden = self.latent_to_hidden(z) # NEW
        
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size) # NEW
        
        decoder_output, _ = self.rnn_rec(inputs, hidden)
        prediction = self.hidden_to_output(decoder_output)
        
        return prediction
    
    
class Decoder_Future(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred, dropout_pred):
        super(Decoder_Future,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_pred
        self.bidirectional = True
        
        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW
        
        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor)
        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0)
        
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        
        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs, hidden)
        
        prediction = self.hidden_to_output(decoder_output)
         
        return prediction


class RNN_VAE(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                        dropout_rec, dropout_pred, softplus):
        super(RNN_VAE,self).__init__()
        
        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
        self.lmbda = Lambda(ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus)
        self.decoder = Decoder(self.seq_len,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred,
                                                 dropout_pred)
        
    def forward(self,seq):
        
        """ Encode input sequence """
        h_n = self.encoder(seq)
        
        """ Compute the latent state via reparametrization trick """
        z, mu, logvar = self.lmbda(h_n)
        ins = z.unsqueeze(2).repeat(1, 1, self.seq_len)
        ins = ins.permute(0,2,1)
        
        """ Predict the future of the sequence from the latent state"""
        prediction = self.decoder(ins, z)
        
        if self.FUTURE_DECODER:
            future = self.decoder_future(ins, z)
            return prediction, future, z, mu, logvar
        else:
            return prediction, z, mu, logvar
        
    def encode(self,seq):
        """ Encode input sequence """
        h_n = self.encoder(seq)
        
        """ Compute the latent state via reparametrization trick """
        z, mu, logvar = self.lmbda(h_n)
        return z, mu, logvar

######
# Training
######


def reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def future_reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def cluster_loss(H, kloss, lmbda, batch_size):
    gram_matrix = (H.T @ H) / batch_size
    _ ,sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:kloss])
    loss = torch.sum(sv)
    return lmbda*loss


def kullback_leibler_loss(mu, logvar):
    # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # I'm using torch.mean() here as the sum() version depends on the size of the latent vector
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def kl_annealing(epoch, kl_start, annealtime, function):
    """
        Annealing of Kullback-Leibler loss to let the model learn first
        the reconstruction of the data before the KL loss term gets introduced.
    """
    if epoch > kl_start:
        if function == 'linear':
            new_weight = min(1, (epoch-kl_start)/(annealtime))

        elif function == 'sigmoid':
            new_weight = float(1/(1+np.exp(-0.9*(epoch-annealtime))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')

        return new_weight

    else:
        new_weight = 0
        return new_weight


def gaussian(ins, is_training, seq_len, std_n=0.8):
    if is_training:
        emp_std = ins.std(1)*std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0,2,1)
        noise = ins.data.new(ins.size()).normal_(0, 1)
        return ins + (noise*emp_std)
    return ins


def train_epoch(train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start,
                annealtime, seq_len, future_decoder, future_steps, scheduler, mse_red, 
                mse_pred, kloss, klmbda, bsize, noise, downsizing_factor = 1):
    model.train() # toggle model to train mode
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)
  
    for idx, data_item in tqdm.tqdm(enumerate(train_loader)):
        
        # data_item = Variable(data_item)
        data_item = data_item.permute(0,2,1)


        data = data_item[:,:seq_len_half,:].type('torch.FloatTensor').to(device)
        fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type('torch.FloatTensor').to(device)
        if noise == True:
            data_gaussian = gaussian(data,True,seq_len_half)
        else:
            data_gaussian = data

        if future_decoder:
            data_tilde, future, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + fut_rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
            fut_loss += fut_rec_loss.item()

        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()
        
        if idx > len(train_loader)//downsizing_factor:
          break
   
    scheduler.step(loss)

    if future_decoder:
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
              mse_loss /idx, fut_loss/idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))
    else:
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
              mse_loss /idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))

    return kl_weight, train_loss/idx, kl_weight*kmeans_losses/idx, BETA*kl_weight*kullback_loss/idx, mse_loss/idx, fut_loss/idx
  
## Data

class SEQUENCE_DATASET(Dataset):
    def __init__(self,data,temporal_window_samples,train):
        self.temporal_window = temporal_window_samples
        self.data = data
        self.data_points = np.shape(self.data)[0]
        
        print('Initialize dataloader. Datapoints %d' %self.data_points)
        
    def __len__(self):        
        return self.data_points

    def __getitem__(self, index):
        temp_window = self.temporal_window 
        start = index-temp_window//2 # temp window was multiplied by 2 at model init
        end = index + temp_window//2
        
        padleft = -min(start, 0)
        padright = max(end - self.data_points, 0)
        
        sequence = self.data[max(start, 0) : min(end, self.data_points), :]
        sequence = np.pad(sequence, ((padleft, padright),(0,0)), mode='reflect')
            
        return torch.from_numpy(sequence.T)
