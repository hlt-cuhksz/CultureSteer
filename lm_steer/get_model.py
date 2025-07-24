
def get_model(model_name, adapted_component, adaptor_class, num_steers, rank,
              epsilon, init_var, steer_type='lora_steer'):
    from model_lora_steer import LoRASteerModule
    model = LoRASteerModule(
        model_name, adapted_component, adaptor_class, num_steers, rank,
        epsilon, init_var)
    return model, model.tokenizer

