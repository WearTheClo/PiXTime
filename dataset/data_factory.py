from dataset.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'electricity': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom
}


def data_provider(args, flag): #[batch size, length, n_vars]
    Data = data_dict[args.data]
    timeenc = 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = 'h'

    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data + '.csv',
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns='Monthly'
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_loader, data_set.enc_in
