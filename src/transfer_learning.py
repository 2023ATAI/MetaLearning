import time
import numpy as np
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn, load_train_data_for_rnn, load_test_data_for_cnn, load_train_data_for_cnn, \
    erath_data_transform, sea_mask_rnn, sea_mask_cnn
from loss import NaNMSELoss
from model import LSTMModel

def train_transfer(x,
          y,
          static,
          static_climate,
          mask, 
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None,
          valid_split=True):
    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split=cfg['valid_split']
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))
    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    if valid_split:
        nt,nf,nlat,nlon = x.shape  #x shape :nt,nf,nlat,nlon
	#Partition validation set and training set
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid, static_valid, climate_valid = x[N:], y[N:], static, static_climate
        x, y = x[:N], y[:N]

    lossmse = torch.nn.MSELoss()
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('static_climate shape is', static_climate.shape)
    print('mask shape is', mask.shape)

    # mask see regions
    #Determine the land boundary
    if cfg['modelname'] in ['LSTM']:
        if valid_split:
            x_valid, y_valid, static_valid, climate_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, climate_valid, mask)
        x, y, static, static_climate = sea_mask_rnn(cfg, x, y, static, static_climate, mask)
    elif cfg['modelname'] in ['CNN','ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)

    # Split data into pre_train and finetune_train sets based on climate types
    pre_train_mask = ((static_climate != 21.0) & (static_climate != 22.0) & (static_climate != 23.0) & (static_climate != 24.0))
    pre_train_indices = np.where(pre_train_mask)
    pre_train_x = x[:, :, pre_train_indices[1]]
    pre_train_y = y[:, pre_train_indices[1]]

    finetune_train_mask = (static_climate == 21.0) | (static_climate == 22.0) | (static_climate == 23.0) | (static_climate == 24.0)
    finetune_train_indices = np.where(finetune_train_mask)
    finetune_train_x = x[:, :, finetune_train_indices[1]]
    finetune_train_y = y[:, finetune_train_indices[1]]


    for num_ in range(cfg['num_repeat']):
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"]*1
            lstmmodel_cfg['out_size'] = 1
            model = LSTMModel(cfg,lstmmodel_cfg).to(device)
        optim = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        # Training phase
        with trange(1, 501) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat)) 
                t_begin = time.time()
                # train
                MSELoss = 0

                for iter in range(0, cfg["niter"]): 
                # ------------------------------------------------------------------------------------------------------------------------------
                #  train way for LSTM model
                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network
                        p_train_x, p_train_y, aux_batch, _, _ = load_train_data_for_rnn(cfg, pre_train_x,
                                                                                        pre_train_y, static,
                                                                                        scaler_y)
                        p_train_x = torch.from_numpy(p_train_x).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        p_train_y = torch.from_numpy(p_train_y).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1,p_train_x.shape[1],1)
                        #print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        #print('x_batch[:,5,0]',x_batch[:,5,0])
                        p_train_x = torch.cat([p_train_x, aux_batch], 2)
                        pred = model(p_train_x, aux_batch)
                        pred = torch.squeeze(pred,1)

                    loss = NaNMSELoss.fit(cfg, pred.float(), p_train_y.float(),lossmse)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    MSELoss += loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                # get loss log
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"], t_end - t_begin)
                print(loss_str)

        # Fine-tuning phase
        with trange(1, cfg['epochs'] + 1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname'] + ' ' + str(num_repeat))  
                t_begin = time.time()
                # train
                MSELoss = 0

                for iter in range(0, cfg["niter"]):  
                    # ------------------------------------------------------------------------------------------------------------------------------
                    #  train way for LSTM model
                    if cfg["modelname"] in \
                            ['LSTM']:
                        # generate batch data for Recurrent Neural Network
                        f_train_x, f_train_y, aux_batch, _, _ = load_train_data_for_rnn(cfg, finetune_train_x,
                                                                                            finetune_train_y, static,
                                                                                            scaler_y)
                        f_train_x = torch.from_numpy(f_train_x).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        f_train_y = torch.from_numpy(f_train_y).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1, f_train_x.shape[1], 1)
                        # print('aux_batch[:,5,0]',aux_batch[:,5,0])
                        # print('x_batch[:,5,0]',x_batch[:,5,0])
                        f_train_x = torch.cat([f_train_x, aux_batch], 2)
                        pred = model(f_train_x, aux_batch)
                        pred = torch.squeeze(pred, 1)

                    loss = NaNMSELoss.fit(cfg, pred.float(), f_train_y.float(), lossmse)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    MSELoss += loss.item()
                # ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                # get loss log
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"],
                                                                               t_end - t_begin)
                print(loss_str)
        # validate
		#Use validation sets to test trained models
		#If the error is smaller than the minimum error, then save the model.
                if valid_split:
                    del f_train_x, f_train_y, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0: 
                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['LSTM']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            n = (x_valid.shape[0]-cfg["seq_len"])//cfg["stride"]
                            for i in range(0, n):
                                #mask
                                x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y,cfg["stride"], i, n)                              
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()

                        t_end = time.time()
                        mse_valid_loss = MSE_valid_loss/(len(gt_list))
                        # get loss log
                        loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss, 
                                    t_end-t_begin)
                        print(loss_str)
                        val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                            wait = 0  # release wait
                            best = val_save_acc #MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
                        wait = 0
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    return
            return


