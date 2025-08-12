import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_PATH = {
            'Llama':"/data/public/Llama/Llama3/Llama3.1/Meta-Llama-3.1-8B",
            "Qwen": "/data/public/Qwen/Qwen2.5/Qwen2.5-7B",
            "CultureLLMLlama": "../extra_models/CultureLLMLlama",
            "CultureLLMQwen": "../extra_models/CultuerLLMQwen",
            "CultureMerge": "../extra_models/CultureMerge",
            "SimLLMLlama":"../extra_models/SimLLMLlama",
            "SimLLMQwen":"../extra_models/SimLLMQwen",
            "CultureSPA":'../extra_models/CultureSPA',
        }


class Config:
    def __init__(self, model_name):
        self.model_name = model_name
        # self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_model(self):
        # 选择对应的模型路径
        model_paths = MODEL_PATH
        if self.model_name not in model_paths:
            if "Llama" in self.model_name:
                self.model_name = "Llama"
            elif "Qwen" in self.model_name:
                self.model_name = "Qwen"
            else:
                raise ValueError(f"模型 '{self.model_name}' 不存在于模型目录！")
        model_path = model_paths[self.model_name]


        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if isinstance(self.tokenizer, bool):
            self.tokenizer = AutoTokenizer.from_pretrained('/data/public/Llama/Llama3/Llama3.1/Meta-Llama-3.1-8B', use_fast=False)
        # 设置 `pad_token_id`
        try:
            if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id + 1  # 确保 `pad_token_id` 和 `eos_token_id` 不同
        except:
            pass
        # 加载模型
        if self.device.type == "cuda":
            print(f"正在使用 CUDA 加载 {model_paths[self.model_name]} 模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            try:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            except:
                pass

            self.model.to(self.device)
        else:
            print("当前使用 CPU，未加载模型，仅加载 tokenizer。")
            self.model = None  # 不加载模型，仅加载 tokenizer
        return self.model, self.tokenizer, self.device
