import torch
import os
from dataset import load_data
from config import Config
from trainer import Trainer, SPMamba


if __name__ == '__main__':
    args = Config().parse_args()
    
    # ---------------------- 消融实验 -----------------------------
    args.dataname = 'pheme'
    args.use_time_enc = True
    mamba_out = False
    use_mask = True
    if args.dataname == 'pheme':
        args.n_class = 2
    
    
    train_iter, dev_iter, test_iter = \
        load_data(args.root_path, args.dataname, args.batch_size, use_time_enc=args.use_time_enc)    
        
    print("Dataset loaded!")    

    model = SPMamba(args.d_input, 
                    args.d_hidden, 
                    args.d_model,
                    args.d_state, 
                    args.q_head, 
                    args.kv_head, 
                    args.n_layer, 
                    args.n_class, 
                    mamba_out)
    
    print("Model initialized!")
    tool = Trainer(model, args.epochs, args.lr, args.device, args.save_path, num_classes=args.n_class, use_mask=use_mask)
    print('Training start ...')
    # tool.train(train_iter, dev_iter)
    
    tool.test(test_iter, args.save_path)
        