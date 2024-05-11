import numpy as np
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import magnav
import psutil
import os

# N = 48 # Length of the time-series
# # Input feature is a sine and a cosine wave
# data_x = np.stack(
#     [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
# )
# data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# # Target output is a sine with double the frequency of the input signal
# data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
# print("data_x.shape: ", str(data_x.shape))
# print("data_y.shape: ", str(data_y.shape))
# data_x = torch.Tensor(data_x)
# data_y = torch.Tensor(data_y)
# dataloader = data.DataLoader(data.TensorDataset(data_x, data_y), batch_size=1, shuffle=True)
# #num_workers=4

def Standard_scaling(df):
    '''
    Apply standardization (Z-score normalization) to a pandas dataframe except for the 'LINE' feature.
    
    Arguments:
    - `df` : dataframe to standardize
    
    Returns:
    - `df_scaled` : standardized dataframe
    '''
    df_scaled = (df-df.mean())/df.std()
    df_scaled['LINE'] = df['LINE']

    return df_scaled

def apply_corrections(df,mags_to_cor,diurnal=True,igrf=True):
    '''
    Apply IGRF and/or diurnal corrections on data.
    
    Arguments:
    - `df` : dataframe to correct #数据集，需要具体航班号 flights[n]
    - `mags_to_cor` : list of string of magnetometers to be corrected # [TL_comp_mag4_cl, TL_com_mag5_cl] 
    - `diurnal` : (optional) apply diunal correction (True or False)
    - `igrf` : (optional) apply IGRF correction (True or False)
    
    Returns:
    - `df_cor` : corrected dataframe
    '''
    mag_measurements = np.array(mags_to_cor)
    df_cor = df.copy()
    
    # Diurnal cor
    if diurnal == True:
        df_cor[mag_measurements] = df_cor[mag_measurements]-np.reshape(df_cor['DIURNAL'].values,[-1,1])
    
    # IGRF cor
    lat  = df_cor['LAT']
    lon  = df_cor['LONG']
    h    = df_cor['BARO']*1e-3
    date = datetime(2020, 6, 29) # Date on which the flights were made
    Be, Bn, Bu = magnav.igrf(lon,lat,h,date)

    if igrf == True:
        df_cor[mag_measurements] = df_cor[mag_measurements]-np.reshape(np.sqrt(Be**2+Bn**2+Bu**2)[0],[-1,1])
        #这一步非常重要，即减去主磁场强度(|B|)，得到磁异常场
    return df_cor

def trim_data(data, seq_len):
    '''
    Delete part of the training data so that the remainder of the Euclidean division 
    between the length of the data and the size of a sequence is 0. 
    This ensures that all sequences are complete.
    
    Arguments:
    - `data` : data that needs to be trimmed
    - `seq_len` : lenght of a sequence
    
    Returns:
    - `data` : trimmed data
    '''
    if (len(data)%seq_len) != 0:
        data = data[:-(len(data)%seq_len)]
    else:
        pass
    
    return data
    
    
class MagNavDataset(Dataset):
    '''
    Transform Pandas dataframe of flights data into a custom PyTorch dataset that returns the data into sequences of a desired length.
    '''
    def __init__(self, df, seq_len, split, train_lines, test_lines,truth='IGRFMAG1'):
        '''
        Initialization of the dataset.
        
        Arguments:
        - `df` : dataframe to transform in a custom PyTorch dataset
        - `seq_len` : length of a sequence
        - `split` : data split ('train' or 'test')
        - `train_lines` : flight lines used for training
        - `test_lines` : flight lines used for testing
        - `truth` : ground truth used as a reference for training the model ('IGRFMAG1' or 'COMPMAG1')
        
        Returns:
        - None
        '''
        self.seq_len  = seq_len
        self.features = df.drop(columns=['LINE',truth]).columns.to_list()
        self.train_sections = train_lines
        self.test_sections = test_lines
        
        if split == 'train':
            
            # Create a mask to keep only training data
            mask_train = pd.Series(dtype=bool)
            for line in self.train_sections:
                mask = (df.LINE == line)
                mask_train = mask|mask_train
            
            # Split in X, y for training
            X_train = df.loc[mask_train,self.features]
            y_train = df.loc[mask_train,truth]
            
            # Trim data and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_train.to_numpy(),dtype=torch.float32),seq_len))
            # print('-------')
            # print(self.X.shape)
            # print('-------')
            self.y = trim_data(torch.tensor(np.reshape(y_train.to_numpy(),[-1,1]),dtype=torch.float32),seq_len)
            
        elif split == 'test':
            
            # Create a mask to keep only testing data
            mask_test = pd.Series(dtype=bool)
            for line in self.test_sections:
                mask = (df.LINE == line)
                mask_test = mask|mask_test
            
            # Split in X, y for testing
            X_test = df.loc[mask_test,self.features]
            y_test = df.loc[mask_test,truth]
            
            # Trim data and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_test.to_numpy(),dtype=torch.float32),seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_test.to_numpy(),[-1,1]),dtype=torch.float32),seq_len)

    def __getitem__(self, idx):
        '''
        Return a sequence for a given index.
        
        Arguments:
        - `idx` : index of a sequence
        
        Returns:
        - `X` : sequence of features
        - `y` : ground truth corresponding to the sequence
        '''
        X = self.X[:,idx:(idx+self.seq_len)]
        y = self.y[idx+self.seq_len-1]
        return X, y
    
    def __len__(self):
        '''
        Return the numbers of sequences in the dataset.
        
        Arguments:
        -None
        
        -Returns:
        -number of sequences in the dataset
        '''
        return len(torch.t(self.X))-self.seq_len

# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr, scaling): #lr=0.005
        super().__init__()
        self.model = model
        self.lr = lr
        self.preds = []
        self.test_running_loss = 0.
        self.Best_error = 9e9
        self.scaling = scaling
        self.log("fold_1003_1007", fold, prog_bar=True)
        self.log("units", units, prog_bar=True)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
#         print("xxxxxxxxxxxxxxx")
#         print(x.shape)
#         print(y.shape)
        y_hat, _ = self.model.forward(x)
#         print(y_hat.shape)
#         print("xxxxxxxxxxxxxxx")
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        #test_running_loss = 0.
        #preds = []
        # print("yyyyyyyyyyyyyy")
        # print(x.shape)
        # print(y.shape)

        with torch.no_grad():
            y_hat, _ = self.model.forward(x)
        #print("yyyyyyyyyyyyyy")
        #print(y_hat.shape)
        #print(y.shape)
        y_hat = y_hat.view_as(y)
        y_hat_t = y_hat*self.scaling[2]+self.scaling[1]
        y_t= y*self.scaling[2]+self.scaling[1]

        loss = nn.MSELoss()(y_hat, y)
        loss2 = magnav.rmse(y_hat_t.cpu(), y_t.cpu(),False)
        self.test_running_loss += loss.item()
        self.preds.append(y_hat.cpu())
        
        # Compute the loss of the batch and save it
        #self.preds = np.concatenate(preds)
        self.log("loss2_mag", loss2, prog_bar=True)
        #print("Best_error{loss2}")

        if self.Best_error > loss2:
            self.Best_error = loss2
            self.log("Best_error", self.Best_error, prog_bar=True)
        #     #Best_model = model

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        # x, y = batch
        # print("zzzzzzzzzzzzzzzz")
        # print(x.shape)
        # print(y.shape)
        # RMSE_epoch = magnav.rmse()(preds*self.scaling[2]+self.scaling[1],test.y[SEQ_LEN:]*self.scaling[2]+self.scaling[1],False)
        # if Best_error > RMSE_epoch:
        #     self.Best_error = RMSE_epoch
        #     self.log("Best_error", self.Best_error, prog_bar=True)
        #     #Best_model = model
        return validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

# Dataloaders
flights = {}
flights_num = [2,3,4,6,7]
df_all = pd.DataFrame()    
for n in flights_num:
    df = pd.read_hdf('../data/Flt_data.h5', key=f'Flt100{n}')
    flights[n] = df
    df_all = pd.concat([df_all,df], ignore_index=True, axis=0)
print(f'Data import done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

train_lines = [np.concatenate([flights[2].LINE.unique(),flights[3].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique()]).tolist(),np.concatenate([flights[2].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique(),flights[7].LINE.unique()]).tolist()]
test_lines  = [flights[7].LINE.unique().tolist(),flights[3].LINE.unique().tolist()]

#----Apply Tolles-Lawson----#
TL = 1
if TL == 1:
    # Get cloverleaf pattern data
    mask = (flights[2].LINE == 1002.20)
    tl_pattern = flights[2][mask]

    # filter parameters
    fs      = 10.0
    lowcut  = 0.1
    highcut = 0.9
    filt    = ['Butterworth',4]
    ridge = 0.025

    for n in tqdm(flights_num): # [2,3,4,6,7]

        # A matrix of Tolles-Lawson
        A = magnav.create_TL_A(flights[n]['FLUXB_X'],flights[n]['FLUXB_Y'],flights[n]['FLUXB_Z'])

        # Tolles Lawson coefficients computation
        TL_coef_2 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG2'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
        TL_coef_3 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG3'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
        TL_coef_4 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG4'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
        TL_coef_5 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG5'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)

        # Magnetometers correction
        flights[n]['TL_comp_mag2_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG2'].tolist(),(-1,1)), TL_coef_2, A)
        flights[n]['TL_comp_mag3_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG3'].tolist(),(-1,1)), TL_coef_3, A)
        flights[n]['TL_comp_mag4_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG4'].tolist(),(-1,1)), TL_coef_4, A)
        flights[n]['TL_comp_mag5_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG5'].tolist(),(-1,1)), TL_coef_5, A)

    print(f'Tolles-Lawson correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

flights_cor = {}
mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl'] #['UNCOMPMAG4', 'UNCOMPMAG5']
for n in tqdm(flights_num):
    flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=True)
del flights

dataset = {}
BATCH_SIZE = 200
SEQ_LEN = 1 #这个需要设置为1
TRUTH='IGRFMAG1'
features = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl',
            'V_BAT1','V_BAT2',
            'INS_VEL_N','INS_VEL_V','INS_VEL_W',
            'CUR_IHTR','CUR_FLAP','CUR_ACLo','CUR_TANK',
            'PITCH','ROLL','AZIMUTH',
            'BARO','LINE',
            #'UTM_X','UTM_Y','UTM_Z',
            #'INS_HGT'
            #'CUR_ACPWR','CUR_OUTPWR',
            #'CUR_BAT1','CUR_BAT2',
            'UNCOMPMAG2','UNCOMPMAG3',
            'TL_comp_mag2_cl','TL_comp_mag3_cl',  
            'UNCOMPMAG4','UNCOMPMAG5',
            'TL_comp_mag4_cl','TL_comp_mag5_cl',    
            #in_f_name,
            TRUTH] 

for n in flights_num:
    dataset[n] = flights_cor[n][features]
del flights_cor
print(f'Feature selection done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

# SCALING == std
SCALING=1
if SCALING == 1:
    # Save scaling parameters
    scaling = {}
    df = pd.DataFrame()
    for flight in flights_num:
        df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
    for n in range(len(test_lines)):
        mask = pd.Series(dtype=bool)
        for line in test_lines[n]:
            temp_mask = (df.LINE == line)
            mask = temp_mask|mask
        scaling[n] = ['std', df.loc[mask,TRUTH].mean(), df.loc[mask,TRUTH].std()]
    del mask, temp_mask, df
    
    # Apply Standard scaling to the dataset
    for n in tqdm(flights_num):
        dataset[n] = Standard_scaling(dataset[n])
    df = pd.DataFrame()
    for flight in flights_num:
        df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
    print(f'Data scaling done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

# Split to train and test
fold = 1 # 0 = 1003, 1 = 1007
train = MagNavDataset(df, seq_len=SEQ_LEN, split='train', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
test  = MagNavDataset(df, seq_len=SEQ_LEN, split='test', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
#print('+++++++++++++________________')
#print(test.y)
#print('+++++++++++++________________')
#print(scaling) #字典, 记录两个数据集的均值，误差 {0: ['std', 8.197226049289801, 233.20539140791215], 1: ['std', -32.82898091607864, 265.09239540180505]}

# Dataloaders
train_loader  = DataLoader(train,
                       batch_size=BATCH_SIZE,
                       shuffle=True,
                       num_workers=0,
                       pin_memory=False)

test_loader    = DataLoader(test,
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=0,
                           pin_memory=False)

# train = MagNavDataset(df_all, seq_len=SEQ_LEN, split='train', train_lines=train_lines[0], test_lines=test_lines[0], truth=TRUTH)
# test  = MagNavDataset(df_all, seq_len=SEQ_LEN, split='test',  train_lines=train_lines[0], test_lines=test_lines[0], truth=TRUTH)

# train_loader  = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True)
# test_loader   = DataLoader(test,batch_size=BATCH_SIZE,shuffle=False)

##====================更换train_loader的列轴顺序================== #在LTC里面已颠倒
##====================矫正 TL, igrf, diurnal, Scaling_STD===========================

# Train the model for 400 epochs (= training steps)

out_features = 1
in_features = len(features)+2

units = 32
wiring = AutoNCP(units, out_features)  # 16 units, 1 motor neuron

ltc_model = LTC(in_features, wiring, batch_first=True)
learn = SequenceLearner(ltc_model, lr=0.002, scaling=scaling[fold])
trainer = pl.Trainer(
    logger=pl.loggers.CSVLogger("log"),
    max_epochs=10,
    gradient_clip_val=1,  # Clip gradient to stabilize training
)

# for batch_index, (inputs, labels) in enumerate(train_loader):
#     # print("==========")
#     # print(batch_index)
#     print("==========")
#     print(inputs.shape)  # (BATCH_SIZE, features, sequences)
#     inputs2 = torch.transpose(inputs, 1, 2)
#     print(inputs2.shape)
#     print("==========")
#     # print(inputs[0,2,:])  # (BATCH_SIZE, features, sequences)
#     # print("==========")
#     # print(labels)
#     print("==========")
#     y = ltc_model(inputs)  #[BATCH_SIZE, Seq, Features]
#     print(y[0].shape) # [BATCH_SIZE, Seq, 1]
#     print("==========")
#     break

trainer.fit(learn, train_loader, test_loader)

# How does the trained model now fit to the sinusoidal function?
#sns.set()
# with torch.no_grad():
#     prediction = ltc_model(data_x)[0].numpy()
# plt.figure(figsize=(6, 4))
# plt.plot(data_y[0, :, 0], label="Target output")
# plt.plot(prediction[0, :, 0], label="NCP output")
# plt.ylim((-1, 1))
# plt.title("After training")
# plt.legend(loc="upper right")
# #plt.show()
# plt.savefig('my_plot.png')
