import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 设置可见的GPU设备
from cal_candidate import main, var_main
from config import Config, MODEL_PATH
from lm_steer.get_model import get_model
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # candidate_related
    parser.add_argument('--model_name',default='Llama',type=str)
    parser.add_argument('--data_dir',default='../../dataset',type=str)
    parser.add_argument('--lang',default='USA',type = str)
    parser.add_argument('--score_dir',default='../results/scores',type=str)
    parser.add_argument('--pwk_dir',default='../results/jsons',type=str)
    parser.add_argument('--baseline',default=False)
    parser.add_argument('--runner', default='cal_s', type=str,help = 'whether to cal candidate p in GPU or cal var in CPU')



    # Training related
    parser.add_argument("--adaptor_class", type=str, default="multiply")
    parser.add_argument("--adapted_component", type=str, default="final_layer")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma_mean", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_length", type=int, default=64) # no need too long
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_step", type=int, default=200)
    parser.add_argument('--train_path',type = str,default='../dataset/train.json')

    # Steer related
    parser.add_argument("--steer_type",type=str,default='lora_steer')
    parser.add_argument("--num_steers", type=int, default=4) # 训练的steer数量, 一个代表向一种风格转换 取4代表四种culture
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--init_var", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=1000) # the rank of projected matrix
    parser

    args = parser.parse_args()

    # set_seed(args.seed)
    if args.runner == 'cal_p':
        model_paths = MODEL_PATH
        ckpt_path = f'../save_model/{args.model_name}/{args.steer_type}.ckpt'
        args = parser.parse_args()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model, tokenizer = get_model(
            model_paths[args.model_name], args.adapted_component, args.adaptor_class,
            args.num_steers,
            args.rank, args.epsilon, args.init_var, args.steer_type)
        model.to_device(device)
        if device.type == 'cpu':

            ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'),weights_only=False)
        else:
            ckpt = torch.load(ckpt_path,weights_only=False)
        model.load_state_dict(ckpt[1],args.steer_type)
        config = Config(args.model_name) # 此时model还是Llama或Qwen
        config.device = torch.device('cpu')
        model_c, tokenizer, device_c = config.select_model()
        config.device = device
        config.model = model
        args.model_name = args.model_name + '_' + f'{args.steer_type}'
        main(args,config)
    elif args.runner == 'cal_s':
        config = Config(args.model_name) # 此时model还是Llama或Qwen
        config.device = torch.device('cpu') # 强制cpu不读取模型
        model, tokenizer, device = config.select_model()
        args.model_name = args.model_name + '_' + f'{args.steer_type}'
        var_main(args,config)

