# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from .steers import Projected_Adaptor
from .model_base import LMSteerBase


class LoRASteerModule(LMSteerBase):
    def __init__(self, model_name, adapted_component, adaptor_class,
                 num_steers, rank, epsilon, init_var):
        super().__init__()
        self.adapted_component = adapted_component

        self.model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.init_var = init_var
        self.num_steers = num_steers
        self.model
        embed_dim = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj','v_proj'],
            lora_dropout=0.05,
            bias='none',
        )
        self.model = get_peft_model(self.model, lora_config)
        for name, _param in self.model.named_parameters():
            if '_lora' not in name:
                _param.requires_grad_ = False
            else:
                _param.requires_grad_ = True

        if adapted_component == "final_layer":
            self.steer = Projected_Adaptor(
                self.model.lm_head, adaptor_class, num_steers, embed_dim,
                vocab_size, rank, epsilon, init_var, "output")
            self.model.set_output_embeddings(self.steer)
            print('set epsilon:', epsilon)
        elif adapted_component == "input_embedding":
            self.steer = Projected_Adaptor(
                self.model.transformer.wte, adaptor_class, num_steers,
                embed_dim, vocab_size, rank, epsilon, init_var, "input")
            self.model.transformer.set_input_embeddings(self.steer)
        else:
            raise NotImplementedError()
