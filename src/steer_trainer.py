import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 设置可见的GPU设备
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import argparse
import json
from lm_steer.utils import set_seed
from lm_steer.get_model import get_model
from lm_steer.utils import RunningMean
from utils import generate
from config import MODEL_PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        text = sample['text']
        ans = sample['label']
        label = sample['lang']
        
        if label == 'USA':
            label = 0
        elif label == 'UK':
            label = 1
        elif label == 'OC':
            label = 2
        elif label == 'CN':
            label = 3
        return text + ans, label


def main(args):
    device = 'cuda' if torch.cuda.is_available() else torch.device("cpu")
    
    train_data = CustomDataset(args.train_path) # 读取data
    dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True)

    model, tokenizer = get_model(
        model_path, args.adapted_component, args.adaptor_class,
        args.num_steers,
        args.rank, args.epsilon, args.init_var, steer_type=args.steer_type)
    model.to(device)  # 确保模型在正确的设备上
    model.train()  # 设置模型为训练模式
    print("number of training epochs:", args.epochs)
    trainable_params = [p for p in model.parameters() if p.requires_grad] + [p for n, p in model.model.named_parameters() if p.requires_grad and 'lora' in n]
    optimizer = Adam(trainable_params, lr=args.lr)

    loss_mean = RunningMean(args.gamma_mean)
    scaler = torch.cuda.amp.GradScaler()
    total_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_text, batch_label in pbar:
            cur_batch_size = len(batch_text)
            batch_stance = torch.zeros(cur_batch_size, args.num_steers).to(device)  # batch_stance 也应放到相同设备

            # batch_stance[torch.arange(batch_stance.shape[0]), batch_label] = 1
            batch_stance[torch.arange(batch_stance.shape[0]).to(device), batch_label.to(device)] = 1  # batch_label应转到正确设备
            if args.dummy_steer is not None:
                batch_stance[:, args.dummy_steer] = 1
            batch_text = batch_text
            tokenized = tokenizer(batch_text, padding=True,
                                  max_length=args.max_length, truncation=True)
            input_ids = torch.LongTensor(tokenized["input_ids"]).to(device)  # input_ids也应放到正确设备

            optimizer.zero_grad()
            attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(device)  # attention_mask 也应转到device
            loss = model(
                input_ids, attention_mask,
                batch_stance.float() # 应该是一个与非门 -- 在第i个元素取1表示对W的第i个维度保留
            ).loss
            # loss = model(
            #     input_ids, attention_mask,
            #     batch_stance.float()
            # ).loss
            loss.backward() # 正则为0
            optimizer.step()
            if total_step % args.save_step == 0:
                torch.save([
                    args, model.state_dict(),
                    total_step
                ], ckpt_name)
                # 生成当前轮次的推理结果 -- generate内部是逐条推理
                batch_prompts = [text[:text.find(':') + 1] for text in batch_text]
                cur_score = generate(batch_prompts,batch_stance,tokenizer,model) # 返回的score暂时没作用 只是print当前的回答
            total_step += 1
            loss_mean.update(loss)
            pbar.set_description(
                f"Epoch {epoch+1}/{args.epochs}: {loss_mean.value}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Llama')
    parser.add_argument("--adaptor_class", type=str, default="multiply")
    parser.add_argument("--adapted_component", type=str, default="final_layer")

    # Training related
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


    # generate related
    parser.add_argument("--temperature", type=int, default=1.0)

    args = parser.parse_args()
    model_paths = MODEL_PATH
    model_path = model_paths[args.model_name]
    ckpt_name = f'../save_model/{args.model_name}/{args.steer_type}.ckpt'
    os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
    set_seed(args.seed)
    main(args)
