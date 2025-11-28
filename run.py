import argparse
import math
import torch
import random
import copy

import numpy as np
import torch.nn as nn

from dataset.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, clever_format
from utils.metrics import metric

import models.DLinear as DLinear
import models.PatchTST as PatchTST
import models.iTransformer as iTransformer
import models.TimeXer as TimeXer
import models.PiXTime as PiXTime
import models.Autoformer as Autoformer

def work_process(args):
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    args.device=torch.device('cuda:'+str(gpu_id))

    train_loader, enc_in  = data_provider(args, flag = 'train')
    val_loader, _ = data_provider(args, flag = 'val')
    test_loader, _ = data_provider(args, flag = 'test')

    if args.model == 'DLinear':
        model = DLinear.Model(seq_len = args.seq_len, pred_len = args.pred_len, enc_in = enc_in).to(args.device)
    elif args.model == 'PatchTST':
        model = PatchTST.Model(seq_len = args.seq_len, pred_len = args.pred_len, patch_len = args.patch_len, d_model = args.d_model,
                               dropout = args.dropout, factor = args.factor, n_heads = args.n_heads, d_ff = args.en_d_ff, e_layers = args.en_layers, enc_in = enc_in).to(args.device)
    elif args.model == 'iTransformer':
        model = iTransformer.Model(seq_len = args.seq_len, pred_len = args.pred_len, d_model = args.d_model, dropout = args.dropout,
                                   factor = args.factor, n_heads = args.n_heads, d_ff = args.en_d_ff, e_layers = args.en_layers, enc_in = enc_in).to(args.device)
    elif args.model == 'TimeXer':
        model = TimeXer.Model(seq_len = args.seq_len, pred_len = args.pred_len, patch_len = args.patch_len, enc_in = enc_in, d_model = args.d_model,
                              dropout = args.dropout, factor = args.factor, n_heads = args.n_heads, d_ff = args.en_d_ff, e_layers = args.en_layers).to(args.device)
    elif args.model == 'PiXTime':
        model = PiXTime.Model(seq_len = args.seq_len, pred_len = args.pred_len, patch_len = args.patch_len, n_vars = enc_in, d_model = args.d_model,
                              dropout = args.dropout, factor = args.factor, n_heads = args.n_heads, en_d_ff = args.en_d_ff, de_d_ff = args.de_d_ff,
                              en_layers = args.en_layers, de_layers = args.de_layers).to(args.device)
    elif args.model == 'Autoformer':
        model = Autoformer.Model(seq_len = args.seq_len, pred_len = args.pred_len, enc_in = enc_in, dec_in = enc_in, c_out = enc_in,
                                 factor = args.factor, e_layers = args.en_layers, d_layers = args.de_layers).to(args.device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): #train
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(args.device) #[batch size * seq_len * channel (S=1)]
            batch_y = batch_y.float().to(args.device) #[batch size * label_len + pred_len * channel (S=1)]
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
            
            loss = loss_func(outputs,batch_y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss = np.average(train_loss)
        
        val_loss = []
        model.eval()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)

            pred = outputs.detach()
            true = batch_y.detach()
                
            loss = loss_func(outputs,batch_y)
            val_loss.append(loss.item())
        val_loss = np.average(val_loss)
        print("Epoch: {0}, Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch+1, train_loss, val_loss))
        adjust_learning_rate(optimizer, epoch + 1, args)

    test_loss = []
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(args.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]
        
            pred = outputs
            true = batch_y
            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    with open(args.evaluation,'a') as f:
        f.write('mse:{}, mae:{}\n'.format(mse, mae))
        f.close()


def run():
    parser = argparse.ArgumentParser(description='Time Series Forecasting.')

    # basic config
    parser.add_argument('--model', type=str, required=False, default='PiXTime',
                        help='model name, options: [DLinear, iTransformer, PatchTST, TimeXer, PiXTime, Autoformer]')
    parser.add_argument('--evaluation', type=str, default='./evaluation/')
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='[ETTh1, ETTm1, ETTh2, ETTm2, electricity, exchange_rate, traffic, weather]')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small/', help='root path of the data file, [dataset/ETT-small/, ./dataset/electricity/, ./dataset/exchange_rate/, ./dataset/traffic/, ./dataset/weather/]')

    # 序列参数
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') #遗存参数，防止报错
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--enc_in', type=int, default=7) #输入数据的频道数，对于ETT，feature=S为1，MS和M是7，在我们的实验中，这个参数由数据集自动给出

    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--en_d_ff', type=int, default=2048, help='dimension of fcn of encoder')
    parser.add_argument('--de_d_ff', type=int, default=2048, help='dimension of fcn of decoder')
    parser.add_argument('--en_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--de_layers', type=int, default=2, help='num of decoder layers')

    # optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    args = parser.parse_args()
    if args.seq_len % args.patch_len != 0 :
        raise ValueError("seq_len无法被patch_len整除")
    file_name  = args.model + args.data + 'PatL' + str(args.patch_len) + 'PreL' + str(args.pred_len) + 'Fea' + args.features + 'DM' + str(args.d_model) + 'ED' + str(args.en_d_ff) + 'DD' + str(args.de_d_ff) + 'EL' + str(args.en_layers) + 'DL' + str(args.de_layers)
    args.evaluation += file_name +'.txt'

    print(args)
    work_process(args)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    fix_seed = 47
    torch.set_num_threads(6)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    run()