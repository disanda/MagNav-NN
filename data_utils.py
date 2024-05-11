#-----------------#
#----Functions----#
#-----------------#

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset   
from datetime import datetime
import magnav 

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

def MinMax_scaling(df, bound=[-1,1]):
    '''
    Apply min-max scaling to a pandas dataframe except for the 'LINE' feature.

    Arguments:
    - `df` : dataframe to standardize
    - `bound` : (optional) upper and lower bound for min-max scaling
    
    Returns:
    - `df_scaled` : scaled dataframe
    '''
    df_scaled = bound[0] + ((bound[1]-bound[0])*(df-df.min())/(df.max()-df.min()))
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