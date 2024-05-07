import time
import numpy as np
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn, load_train_data_for_rnn, load_test_data_for_cnn, load_train_data_for_cnn, \
    erath_data_transform, sea_mask_rnn, sea_mask_cnn, _get_k_shot, _grad_step
from loss import NaNMSELoss
import learn2learn as l2l
from model import LSTMModel, CNN, ConvLSTMModel


def train_meta(x,
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
    # Initialize training parameters
    patience = cfg['patience']
    wait = 0
    best = 9999
    # valid_split = cfg['valid_split']
    # Print device and data types for verification
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))

    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
        #	Splice x according to the sphere shape
        lat_index, lon_index = erath_data_transform(cfg, x)
        print(
            '\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(
                m_n=cfg['modelname']))
    if valid_split:
        nt, nf, nlat, nlon = x.shape  # x shape :nt,nf,nlat,nlon
        # Partition validation set and training set
        N = int(nt * cfg['split_ratio'])
        x_valid, y_valid, static_valid, climate_valid = x[N:], y[N:], static, static_climate
        x, y = x[:N], y[:N]

    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('static_climate shape is', static_climate.shape)
    print('mask shape is', mask.shape)

    lossmse = torch.nn.MSELoss(reduction="mean")

    # mask see regions
    # Determine the land boundary
    if cfg['modelname'] in ['meta-LSTM']:
        if valid_split:
            x_valid, y_valid, static_valid, climate_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid,
                                                                         climate_valid, mask)
        x, y, static, static_climate = sea_mask_rnn(cfg, x, y, static, static_climate, mask)
    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)

    # Split data into base_train, base_test, target_train, and target_test sets
    base_train_mask = (static_climate != 29.0)
    base_train_indices = np.where(base_train_mask)
    base_train_x = x[:, :, base_train_indices[1]]
    base_train_y = y[:, base_train_indices[1]]

    
    base_test_mask = (climate_valid != 29.0)
    base_test_indices = np.where(base_test_mask)
    base_test_x = x_valid[:, :, base_test_indices[1]]
    base_test_y = y_valid[:, base_test_indices[1]]

    
    target_train_mask = (static_climate == 29.0)
    target_train_indices = np.where(target_train_mask)
    target_train_x = x[:, :, target_train_indices[1]]
    target_train_y = y[:, target_train_indices[1]]

    
    target_test_mask = (climate_valid == 29.0)
    target_test_indices = np.where(target_test_mask)
    target_test_x = x_valid[:, :, target_test_indices[1]]
    target_test_y = y_valid[:, target_test_indices[1]]

    # prepare models
    # Selection model
    if cfg['modelname'] in ['meta-LSTM']:
        lstmmodel_cfg = {}
        lstmmodel_cfg['input_size'] = cfg["input_size"]
        lstmmodel_cfg['hidden_size'] = cfg["hidden_size"] * 1
        lstmmodel_cfg['out_size'] = 1
        lstmmodel_cfg['num_layers'] = 2
        lstmmodel_cfg['num_heads'] = 2
        model = LSTMModel(cfg,lstmmodel_cfg).to(device)

    elif cfg['modelname'] in ['CNN']:
        model = CNN(cfg).to(device)
    elif cfg['modelname'] in ['ConvLSTM']:
        model = ConvLSTMModel(cfg).to(device)

    # Initialize MAML algorithm
    maml = l2l.algorithms.MAML(model, lr=0.001, first_order=False)
    optim1 = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    optim2 = torch.optim.Adam(maml.parameters(), lr=cfg['learning_rate'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=cfg['epochs'])

    # Main training loop
    for num_ in range(cfg['num_repeat']):
        with trange(1, cfg['epochs'] + 1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname'] + ' ' + str(num_repeat))  
                t_begin = time.time()
                base_train_loss, target_train_error = 0.0, 0.0
                model = maml.module

                if cfg["modelname"] in ['meta-LSTM']:
                    # Inner-loop optimization for base_train data
                    for iter in range(0, cfg["niter1"]):  

                        # Propose phi using base sets
                        b_train_x, b_train_y, aux_batch, _, _ = load_train_data_for_rnn(cfg, base_train_x,
                                                                                        base_train_y, static,
                                                                                        scaler_y)
                        b_train_x = torch.from_numpy(b_train_x).to(device)
                        b_train_y = torch.from_numpy(b_train_y).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)

                        # Inner-loop to propose phi using meta-training dataset
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1, b_train_x.shape[1], 1)
                        b_train_x = torch.cat([b_train_x, aux_batch], 2)
                        pred = model(b_train_x, aux_batch)
                        pred = torch.squeeze(pred, 1)
                        loss = NaNMSELoss.fit(cfg, pred.float(), b_train_y.float(), lossmse)
                        optim1.zero_grad()
                        loss.backward()
                        optim1.step()
                        base_train_loss += loss

                    b_train_loss = base_train_loss / cfg['niter1']

                    del b_train_x, b_train_y, aux_batch
                    # Outer-loop optimization for target_train data
                    for iter in range(0, cfg["niter2"]):  


                        # Outer-loop using target sets
                        t_train_x, t_train_y, aux_batch, _, _ = load_train_data_for_rnn(cfg, target_train_x,
                                                                                        target_train_y, static,
                                                                                        scaler_y)
                        t_train_x = torch.from_numpy(t_train_x).to(device)
                        t_train_y = torch.from_numpy(t_train_y).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)

                        # accumulate inner-loop gradients given proposed phi
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1, t_train_x.shape[1], 1)
                        t_train_x = torch.cat([t_train_x, aux_batch], 2)
                        pred = model(t_train_x, aux_batch)
                        pred = torch.squeeze(pred, 1)
                        loss = NaNMSELoss.fit(cfg, pred.float(), t_train_y.float(), lossmse)
                        target_train_error += loss

                    t_train_loss = target_train_error / cfg['niter2']
                    del t_train_x, t_train_y, aux_batch



                # Parameter outer-loop update
                optim2.zero_grad()
                target_train_error.backward()
                _grad_step(maml, iter, optim2, schedule)

                if cfg["modelname"] in ['meta-LSTM']:
                    if valid_split:
                        base_test_loss = 0.0
                        target_test_loss = 0.0
                        if epoch % 20 == 0:  
                            # NOTE: We used grids-mean NSE as valid metrics.

                            gt_list1 = [i for i in range(0, base_test_x.shape[0] - cfg['seq_len'], cfg["stride"])]
                            n1 = (base_test_x.shape[0] - cfg["seq_len"]) // cfg["stride"]
                            for i in range(0, n1):
                                b_test_x, b_test_y, aux_batch, _, _ = load_test_data_for_rnn(cfg, base_test_x,
                                                                                             base_test_y,
                                                                                             static, scaler_y,
                                                                                             cfg["stride"], i, n1)
                                b_test_x = torch.Tensor(b_test_x).to(device)
                                b_test_y = torch.Tensor(b_test_y).to(device)
                                aux_batch = torch.Tensor(aux_batch).to(device)
                                aux_batch = aux_batch.unsqueeze(1)
                                aux_batch = aux_batch.repeat(1, b_test_x.shape[1], 1)
                                b_test_x = torch.cat([b_test_x, aux_batch], 2)
                                with torch.no_grad():
                                    pred = model(b_test_x, aux_batch)
                                loss_b = NaNMSELoss.fit(cfg, pred, b_test_y, lossmse)
                                base_test_loss += loss_b.item()

                            b_test_loss = base_test_loss / (len(gt_list1))

                            gt_list2 = [i for i in range(0, target_test_x.shape[0] - cfg['seq_len'], cfg["stride"])]
                            n2 = (target_test_x.shape[0] - cfg["seq_len"]) // cfg["stride"]
                            for i in range(0, n2):
                                t_test_x, t_test_y, aux_batch, _, _ = load_test_data_for_rnn(cfg, target_test_x,
                                                                                             target_test_y,
                                                                                             static, scaler_y,
                                                                                             cfg["stride"], i, n2)
                                t_test_x = torch.Tensor(t_test_x).to(device)
                                t_test_y = torch.Tensor(t_test_y).to(device)
                                aux_batch = torch.Tensor(aux_batch).to(device)
                                aux_batch = aux_batch.unsqueeze(1)
                                aux_batch = aux_batch.repeat(1, t_test_x.shape[1], 1)
                                t_test_x = torch.cat([t_test_x, aux_batch], 2)
                                with torch.no_grad():
                                    pred = model(t_test_x, aux_batch)
                                loss_t = NaNMSELoss.fit(cfg, pred, t_test_y, lossmse)
                                target_test_loss += loss_t.item()

                            t_test_loss = target_test_loss / (len(gt_list2))

                            val_save_acc = b_test_loss*0.8 + t_test_loss*0.2

                            if val_save_acc < best:
                                # if MSE_valid_loss < best:
                                torch.save(maml, out_path + cfg['modelname'] + '_para.pkl')
                                wait = 0  # release wait
                                best = val_save_acc  # MSE_valid_loss
                                print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                t_end = time.time()
                if epoch % 20 == 0:
                    loss_str = "Epoch {}, Base_test_error {:.3f}, Target_test_error {:.3f}, time {:.2f}".format(epoch,  b_test_loss,  t_test_loss, t_end - t_begin)
                    print('\033[91m' + loss_str + '\033[0m')
                else:
                    loss_str = "Epoch {}, Base_train_error {:.3f}, Target_train_error {:.3f}, time {:.2f}".format(epoch, b_train_loss, t_train_loss, t_end - t_begin)
                    print(loss_str)
