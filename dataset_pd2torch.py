import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.CNN import Optuna_CNN, ResNet18, ResNet18v2
from src.models.RNN import LSTM, GRU
from src.models.MLP import MLP

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


# Dataloaders
BATCH_SIZE = 5
SEQ_LEN = 20
TRUTH='IGRFMAG1'

flights = {}
flights_num = [2,3,4,6,7]


df_all = pd.DataFrame()    
for n in flights_num:
    df = pd.read_hdf('./data/Flt_data.h5', key=f'Flt100{n}')
    flights[n] = df
    df_all = pd.concat([df_all,df], ignore_index=True, axis=0)

# mags_to_cor = ['UNCOMPMAG4', 'UNCOMPMAG5']
# features = [mags_to_cor[0],mags_to_cor[1], 
#                 'V_BAT1','V_BAT2','INS_VEL_N','INS_VEL_V',
#                 'INS_VEL_W','CUR_IHTR', 'CUR_FLAP','CUR_ACLo',
#                 'CUR_TANK','PITCH', 'ROLL','AZIMUTH',
#                 'BARO','LINE',
#                 TRUTH] # len = 17
# for n in flights_num:
#     dataset[n] = flights_cor[n][features]
#df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)

train_lines = [np.concatenate([flights[2].LINE.unique(),flights[3].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique()]).tolist(),
    np.concatenate([flights[2].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique(),flights[7].LINE.unique()]).tolist()]
test_lines  = [flights[7].LINE.unique().tolist(),flights[3].LINE.unique().tolist()]

train = MagNavDataset(df_all, seq_len=SEQ_LEN, split='train', train_lines=train_lines[0], test_lines=test_lines[0], truth=TRUTH)
test  = MagNavDataset(df_all, seq_len=SEQ_LEN, split='test',  train_lines=train_lines[0], test_lines=test_lines[0], truth=TRUTH)

train_loader  = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True)
#test_loader   = DataLoader(test,batch_size=BATCH_SIZE,shuffle=False)

# Always keep the 'LINE' feature in the feature list so that the MagNavDataset function can split the flight data
features = ['UNCOMPMAG4', 'UNCOMPMAG5','V_BAT1','V_BAT2',
                    'INS_VEL_N','INS_VEL_V','INS_VEL_W','CUR_IHTR',
                    'CUR_FLAP','CUR_ACLo','CUR_TANK','PITCH',
                    'ROLL','AZIMUTH','BARO','LINE',TRUTH] # 15 features

#model = Optuna_CNN(SEQ_LEN,96) # 96 features
#conv1 = torch.nn.Conv1d(in_channels = 96, out_channels = 16, kernel_size=3, stride =1, padding =1, padding_mode='zeros')
#max_pool1 = torch.nn.MaxPool1d(kernel_size=2,stride=2)
#conv2 = torch.nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size=3, stride =1, padding =1, padding_mode='zeros')
#max_pool2 = torch.nn.MaxPool1d(kernel_size=2,stride=2)

num_LSTM    = 2
hidden_size = [32,32]
num_layers  = [3,1]
num_linear  = 2
num_neurons = [16,4]
#model = LSTM(SEQ_LEN, hidden_size, num_layers, num_LSTM, num_linear, num_neurons, 'cpu')

model = GRU(SEQ_LEN,33,3,'cpu') # 20 * 15 = 300

for batch_index, (inputs, labels) in enumerate(train_loader):
    # print("==========")
    # print(batch_index)
    # print("==========")
    # print(inputs.shape)  # (BATCH_SIZE, features, sequences)
    # print("==========")
    # print(inputs[0,2,:])  # (BATCH_SIZE, features, sequences)
    # print("==========")
    # print(labels)
    print("==========")
    print(model(inputs))
    print("==========")

    # y = conv1(inputs)
    # print(y.shape) # [BATCH_SIZE, out_channels, SEQ_LEN]

    # y2 = max_pool1(y)
    # print(y2.shape)  # [BATCH_SIZE, out_channels, SEQ_LEN//2]

    # y3 = conv2(y2)
    # print(y3.shape)

    # y4 = max_pool2(y3)
    # print(y4.shape)

    break





