from data_provider.data_loader import Dataset_DOE_hour
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_DOE_hour
    timeenc = 0 if args.embed != 'timeF' else 1

    # For testing, shuffle is turned to False.
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Creating the data batch based on flag -> Pred flag removed now.
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len, args.interval_len],
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))

    # Creating the dataloader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True)

    return data_set, data_loader
