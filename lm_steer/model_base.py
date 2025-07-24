
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from lm_steer.utils import set_seed
from .model_utils import find_max_subspans


punctuations = [
    '!', '"', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
    # '/', '#',
    ':', ';', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`',
    '{', '|', '}', '~',
    '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·',
    '¸', '¹', 'º', '»', '¼', '½', '¾',
    '\n', ' ',
]


class LMSteerBase(nn.Module):    
    # 在这里也要修改
    def generate(self, input_ids, attention_mask, steer_values, seed, max_new_tokens=5, output_scores=True,
                 temperature=0.8, return_dict_in_generate=True, device='cuda'):
        self.to_device(device)
        if seed is not None:
            set_seed(seed)
        _steer_values = torch.zeros([1,4]).to(self.device) # 因为是一条一条推理 所以设置为1,4
        _steer_values[:,steer_values] = 1
        _steer_values = torch.Tensor(_steer_values).to(
            self.device)
        self.steer.set_value(_steer_values)
        with torch.no_grad():

            output = self.model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        max_new_tokens=max_new_tokens, 
                        output_scores=output_scores,
                        temperature = temperature,
                        do_sample=True,
                        top_p = 0.95,
                        return_dict_in_generate=return_dict_in_generate
                    )
        text = self.tokenizer.decode(output[0].squeeze(0), skip_special_tokens=True)
        scores = output['scores']
        scores = torch.stack(scores, dim=0)
        return text, scores # 输出score

    def state_dict(self):
        lora_param = {}
        for name, param in self.named_parameters():
            if 'lora' in name:
                lora_param[name] = param
        
        return {**lora_param,**self.steer.state_dict()}

    def load_state_dict(self, state_dict, steer_type):
        lora_state_dict = {'.'.join((str(k).split('.'))[1:]): v for k, v in state_dict.items() if 'lora_' in k}
        missing, unexpected = self.model.load_state_dict(lora_state_dict, strict=False)
        self.steer.load_state_dict(state_dict) # 读取steer参数

    def parameters(self):
        return self.steer.parameters()

    def to_device(self, device):
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        return self.steer.regularization_term()

    def forward(self, input_ids, attention_mask, steer_values):
        self.steer.set_value(steer_values)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids) 
        return output
