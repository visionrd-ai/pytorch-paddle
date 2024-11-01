import json
import torch
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_weights(model, path, logger=None):
    if path:
        def_dict = torch.load(path)
        model.load_state_dict(def_dict)
        if logger:
            logger.info(f"Loaded model weights from {path}")
        return model
    return model