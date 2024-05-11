#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

import argparse
import warnings
import os
from datetime import datetime
import time
import math
import psutil

import magnav
from models.CNN import Optuna_CNN, ResNet18, ResNet18v2
from models.RNN import LSTM, GRU
from models.MLP import MLP
from ncps.torch import CfC

import data_utils
from data_utils import trim_data, Standard_scaling, MinMax_scaling, apply_corrections


def make_training(model, epochs, train_loader, test_loader, scaling=['None']):
    '''
    PyTorch training loop with testing.
    
    Arguments:
    - `model` : model to train
    - `epochs` : number of epochs to train the model
    - `train_loader` : PyTorch dataloader for training
    - `test_loader` : PyTorch dataloader for testing
    - `scaling` : (optional) scaling parameters
    
    Returns:
    - `train_loss_history` : history of loss values during training
    - `test_loss_history` : history of loss values during testing
    '''
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
    lambda1 = lambda epoch: 0.9**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    # Create batch and epoch progress bar
    batch_bar = tqdm(total=len(train)//BATCH_SIZE,unit="batch",desc='Training',leave=False, position=0, ncols=150)
    epoch_bar = tqdm(total=epochs,unit="epoch",desc='Training',leave=False, position=1, ncols=150)
    
    train_loss_history = []
    test_loss_history = []
    Best_RMSE = 9e9

    for epoch in range(epochs):

        #----Train----#

        train_running_loss = 0.

        # Turn on gradients computation
        model.train()
        
        batch_bar.reset()
        
        # Enumerate allow to track batch index and intra-epoch reporting 
        for batch_index, (inputs, labels) in enumerate(train_loader):
            #print('******************')
            #print(batch_index)

            # Put data to the desired device (CPU or GPU)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero gradients of optimizer for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            if model.__class__.__name__ != 'CfC':
                predictions = model(inputs)
                #print('------------------------')
            else:
                inputs = inputs.transpose(2,1) # (batch_size, seq, in_features) > (batch_size, in_features, seq)
                if batch_index == 0:
                    h0 = torch.zeros(inputs.size()[0],1).to(DEVICE)
                    predictions, h1 = model(inputs,h0) # (batch_size, seq, out_features)/(batch_size, out_features)
                    #print('=======================')
                else:
                    predictions, h1 = model(inputs,h1[-inputs.size()[0]:].detach())
                    #print('########################')
                predictions = predictions[:,-1,:]

            # Compute the loss
            loss = criterion(predictions, labels)
            #print(predictions.shape)
            #print(labels.shape)

            # Calculate gradients
            loss.backward() # retain_graph=True

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_running_loss += loss.item()
            
            # Update batch progess bar
            batch_bar.set_postfix(train_loss=train_running_loss/(batch_index+1),lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()
        
        # Update learning rate
        scheduler.step()

        # Compute the loss of the batch and save it
        train_loss = train_running_loss / batch_index
        train_loss_history.append(train_loss)

        #----Test----#

        test_running_loss = 0.
        preds = []
        
        # Disable layers specific to training such as Dropout/BatchNorm
        model.eval()
        
        # Turn off gradients computation
        with torch.no_grad():
            
            # Enumerate allow to track batch index and intra-epoch reporting
            for batch_index, (inputs, labels) in enumerate(test_loader):

                # Put data to the desired device (CPU or GPU)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Make prediction for this batch
                #predictions = model(inputs)

                # Make predictions for this batch
                if model.__class__.__name__ != 'CfC':
                    predictions = model(inputs)
                    #print('------------------------')
                else:
                    inputs = inputs.transpose(2,1) # (batch_size, seq, in_features) > (batch_size, in_features, seq)
                    if batch_index == 0:
                        h0 = torch.zeros(inputs.size()[0],1).to(DEVICE)
                        predictions, h1 = model(inputs,h0) # (batch_size, seq, out_features)/(batch_size, out_features)
                        #print('=======================')
                    else:
                        predictions, h1 = model(inputs,h1[-inputs.size()[0]:].detach())
                        #print('########################')
                    predictions = predictions[:,-1,:]
                
                # Save prediction for this batch
                preds.append(predictions.cpu())

                # Compute the loss
                loss = criterion(predictions, labels)

                # Gather data and report
                test_running_loss += loss.item()

        # Compute the loss of the batch and save it
        preds = np.concatenate(preds)

        if scaling[0] == 'None':
            RMSE_epoch = magnav.rmse(preds,test.y[SEQ_LEN:],False)
        elif scaling[0] == 'std':
            RMSE_epoch = magnav.rmse(preds*scaling[2]+scaling[1],test.y[SEQ_LEN:]*scaling[2]+scaling[1],False)
        elif scaling[0] == 'minmax':
            RMSE_epoch = magnav.rmse(scaling[3]+((preds-scaling[1])*(scaling[4]-scaling[3])/(scaling[2]-scaling[1])),
                               scaling[3]+((test.y[SEQ_LEN:]-scaling[1])*(scaling[4]-scaling[3])/(scaling[2]-scaling[1])),False)

        test_loss = test_running_loss / batch_index
        test_loss_history.append(test_loss)
        
        # Save best model
        if Best_RMSE > RMSE_epoch:
            Best_RMSE = RMSE_epoch
            Best_model = model

        # Update epoch progress bar
        epoch_bar.set_postfix(train_loss=train_loss,test_loss=test_loss,RMSE=RMSE_epoch,lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    print('\n')
    
    return train_loss_history, test_loss_history, Best_RMSE, Best_model


#------------#
#----Main----#
#------------#


if __name__ == "__main__":
    
    # Start timer
    start_time = time.time()
    
    # set seed for reproducibility
    seed = 28 # 27
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    #----User arguments----#
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d","--device", type=str, required=False, default='cuda', help="Which GPU to use (cuda or cpu), default='cuda'. Ex : --device 'cuda' ", metavar=""
    )
    parser.add_argument(
        "-e","--epochs", type=int, required=False, default=35, help="Number of epochs to train the model, default=35. Ex : --epochs 200", metavar=""
    )
    parser.add_argument(
        "-b","--batch", type=int, required=False, default=256, help="Batch size for training, default=256. Ex : --batch 64", metavar=""
    )
    parser.add_argument(
        "-sq","--seq", type=int, required=False, default=1, help="Length sequence of data, default=20. Ex : --seq 15", metavar=""
    )
    parser.add_argument(
        "--shut", action="store_true", required=False, help="Shutdown pc after training is done."
    )
    parser.add_argument(
        "-sc", "--scaling", type=int, required=False, default=1, help="Data scaling, 1 for standardization, 2 for MinMax scaling, 0 for no scaling, default=0. Ex : --scaling 0", metavar=''
    )
    parser.add_argument(
        "-cor", "--corrections", type=int, required=False, default=3, help="Data correction, 0 for no corrections, 1 for IGRF correction, 2 for diurnal correction, 3 for IGRF+diurnal correction. Ex : --corrections 3", metavar=''
    )
    parser.add_argument(
        "-tl", "--tolleslawson", type=int, required=False, default=1, help="Apply Tolles-Lawson compensation to data, 0 for no compensation, 1 for compensation. Ex : --tolleslawson 1", metavar=''
    )
    parser.add_argument(
        "-tr", "--truth", type=str, required=False, default='IGRFMAG1', help="Name of the variable corresponding to the truth for training the model. Ex : --truth 'IGRFMAG1'", metavar=''
    )
    parser.add_argument(
        "-ml", "--model", type=str, required=False, default='CfC', help="Name of the model to use. Available models : 'MLP', 'CNN', 'ResNet18', 'LSTM', 'GRU', 'ResNet18v2', 'MLP_3L, MLP_3Lv2'. Ex : --model 'CNN'", metavar=''
    )
    parser.add_argument(
        "-wd", "--weight_decay", type=float, required=False, default=0.001, help="Adam weight decay value. Ex : --weight_decay 0.00001", metavar=''
    )
    
    args = parser.parse_args()
    in_f_name = 'FLUXCD_X_Y_Z_TOT-D_noBat1' #'CUR_STRB',
    shuffle_= True
    hidden_features =4096 #MLP
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    DEVICE     = args.device
    SEQ_LEN    = args.seq
    SCALING    = args.scaling
    COR        = args.corrections # 3
    TL         = args.tolleslawson # 1
    TRUTH      = args.truth
    MODEL      = args.model
    WEIGHT_DECAY = args.weight_decay
    
    if DEVICE == 'cuda':
        print(f'\nCurrently training on {torch.cuda.get_device_name(DEVICE)}')
    else:
        print('Currently training on cpu.')

    #----Import data----#
    
    flights = {}
    
    # Flights to import
    flights_num = [2,3,4,6,7]
    
    for n in flights_num:
        df = pd.read_hdf('../Flt_data.h5', key=f'Flt100{n}')
        flights[n] = df
    
    print(f'Data import done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Slecting train/test lines----#
    
    train_lines = [np.concatenate([flights[2].LINE.unique(),flights[3].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique()]).tolist(),
                   np.concatenate([flights[2].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique(),flights[7].LINE.unique()]).tolist()]
    test_lines  = [flights[7].LINE.unique().tolist(),
                   flights[3].LINE.unique().tolist()]
    
    #----Apply Tolles-Lawson----#
    
    # Get cloverleaf pattern data
    mask = (flights[2].LINE == 1002.20)
    tl_pattern = flights[2][mask]

    # filter parameters
    fs      = 10.0
    lowcut  = 0.1
    highcut = 0.9
    filt    = ['Butterworth',4]
    
    ridge = 0.025
    if TL == 1:
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
    else:
        print(f'Tolles-Lawson correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

    #----Apply IGRF and diurnal corrections----#

    flights_cor = {}

    if TL == 0:
        mags_to_cor = ['UNCOMPMAG4', 'UNCOMPMAG5']
        mags_to_cor2 = []
    if TL == 1:
        mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']
        mags_to_cor2 = []
    if TL == 2:
        mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']
        mags_to_cor2 = ['UNCOMPMAG4', 'UNCOMPMAG5']
    
    if COR == 0:
        flights_cor = flights.copy()
        del flights
        print(f'No correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    if COR == 1:
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=False, igrf=True)
        del flights
        print(f'IGRF correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    if COR == 2: 
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=False)
        del flights
        print(f'Diurnal correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    if COR == 3:
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=True)
        del flights
        print(f'IGRF+Diurnal correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Select features----#
    
    # Always keep the 'LINE' feature in the feature list so that the MagNavDataset function can split the flight data
    other_features = [#'V_BAT1',
                      'V_BAT2',
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
                'FLUXC_X','FLUXC_Y','FLUXC_Z',#'FLUXC_TOT',
                'FLUXD_X','FLUXD_Y','FLUXD_Z','FLUXD_TOT',
                #in_f_name,
                TRUTH]
    features = mags_to_cor + mags_to_cor2 + other_features 
    
    dataset = {}
    
    for n in flights_num:
        dataset[n] = flights_cor[n][features]
    
    del flights_cor
    print(f'Feature selection done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Data scaling----#
    
    if SCALING == 0:
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
        
        # Save scaling parameters
        scaling = {}
        for n in range(len(test_lines)):
            scaling[n] = ['None']
        
    elif SCALING == 2:
        # Save scaling parameters
        bound = [-1,1]
        scaling = {}
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
        for n in range(len(test_lines)):
            mask = pd.Series(dtype=bool)
            for line in test_lines[n]:
                temp_mask = (df.LINE == line)
                mask = temp_mask|mask
            scaling[n] = ['minmax', bound[0], bound[1], df.loc[mask,TRUTH].min(), df.loc[mask,TRUTH].max()]
        del mask, temp_mask, df
        
        # Apply Min-Max sacling to the dataset
        for n in tqdm(flights_num):
            dataset[n] = MinMax_scaling(dataset[n], bound=bound)
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)

    elif SCALING == 1:
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
    
    del dataset
    print(f'Data scaling done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Training----#
    
    train_loss_history = []
    test_loss_history = []
    RMSE_history = []
    
    # Cross validation with selected folds 
    for fold in range(len(train_lines)): # 2
        
        print('\n--------------------')
        print(f'Fold number {fold}')
        print('--------------------\n')
        
        # Split to train and test
        train = data_utils.MagNavDataset(df, seq_len=SEQ_LEN, split='train', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
        test  = data_utils.MagNavDataset(df, seq_len=SEQ_LEN, split='test', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)

        # Dataloaders
        train_loader  = DataLoader(train,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=False) #drop_last=True

        test_loader    = DataLoader(test,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False) #drop_last=True

        # Model
        if MODEL == 'MLP':
            model = MLP(SEQ_LEN,len(features)-2, hidden_features).to(DEVICE)
        elif MODEL == 'MLP_3L':
            model = MLP_3L(SEQ_LEN,len(features)-2,hidden_features).to(DEVICE)
        elif MODEL == 'MLP_3Lv2':
            model = MLP_3Lv2(SEQ_LEN,len(features)-2,hidden_features).to(DEVICE)
        elif MODEL == 'CNN':
            filters=[32,64]
            num_neurons=[64,32]
            n_convblock=2
            model = Optuna_CNN(SEQ_LEN,len(features)-2,n_convblock,filters,num_neurons).to(DEVICE)
        elif MODEL == 'ResNet18':
            model = ResNet18().to(DEVICE)
        elif MODEL == 'ResNet18v2':
            model = ResNet18v2().to(DEVICE)
        elif MODEL == 'LSTM':
            num_LSTM    = 1
            hidden_size = [32]
            num_layers  = [1]
            num_linear  = 1
            num_neurons = [32]            
            model = LSTM(SEQ_LEN, hidden_size, num_layers, num_LSTM, num_linear, num_neurons, DEVICE).to(DEVICE)
        elif MODEL == 'GRU':
            num_layers = 11
            model = GRU(SEQ_LEN,len(features)-2,num_layers = 11).to(DEVICE) # 20 * 15 = 300
        elif MODEL == 'CfC':
            model = CfC(len(features)-2,1).to(DEVICE) 
        model.name = model.__class__.__name__

        # Loss function
        criterion = torch.nn.MSELoss()

        # Training
        train_loss, test_loss, Best_RMSE, Best_model = make_training(model, EPOCHS, train_loader, test_loader, scaling[fold])
        
        # Save results from training
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        RMSE_history.append(Best_RMSE)
        
        if fold == 0:
            folder_path = f'experiments/=HF_{hidden_features}_{Best_model.name}_{scaling[0][0]}_TL{TL}_COR{COR}_{TRUTH}'
            #os.mkdir(folder_path)
            os.makedirs(folder_path, exist_ok=True)
        torch.save(Best_model,folder_path+f'/{Best_model.name}_fold{fold}.pt')
    
    # Compute pre-processing+training time
    end_time = time.time()-start_time
    
    # Compute global perf over all folds
    perf_folds = sum(RMSE_history)/len(train_lines)
    
    # Print perf of training
    print('\n-------------------------')
    print('Performance for all folds')
    print('-------------------------')

    for n in range(len(test_lines)):
        print(f'Fold {n} | RMSE = {RMSE_history[n]:.2f} nT')
    print(f'Total  | RMSE = {perf_folds:.2f} nT')
    now = datetime.now()
    now_time = now.strftime('%Y-%m-%d_%H:%M:%S')
    print(folder_path+now_time)
    
    # Show performance graphs
    for fold in range(len(test_lines)):
        fig, ax = plt.subplots(figsize=[10,4])
        ax.plot(train_loss_history[fold], label='Train loss')
        ax.plot(test_loss_history[fold], label='Test loss')
        ax.set_title(f'Loss for fold {fold}')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.savefig(folder_path+f'/losses_fold{fold}')
    
    # Save parameters
    params = pd.DataFrame(columns=['seq_len','epochs','batch_size','training_time','now_time','fold0','fold1','fold_total'])
    params.loc[0,'seq_len'] = SEQ_LEN
    params.loc[0,'epochs'] = EPOCHS
    params.loc[0,'batch_size'] = BATCH_SIZE
    params.loc[0,'training_time'] = end_time
    params.loc[0,'now_time'] = now_time
    params.loc[0,'fold0'] = RMSE_history[0]
    params.loc[0,'fold1'] = RMSE_history[1]
    params.loc[0,'fold_total'] = perf_folds
    params.to_csv(folder_path+f'/parameters.csv', index=False)
    
    # Empty GPU ram
    torch.cuda.empty_cache()
    
    # Shutdown computer
    if args.shut == True:
        os.system("shutdown")