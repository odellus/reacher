# -*- coding: utf-8

import yaml
import torch
import pickle as pkl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device  = torch.device("cpu")

def load_cfg(filepath="./config.yaml"):
    """Load YAML configuration file

    Params
    ======
    filepath (str): The path of the YAML configuration file
    """
    with open(filepath, "r") as f:
        return yaml.load(f)

def get_state_action_sizes(env):
    """Get the state and action space dimensions of a Gym environment

    Params
    ======
    env (gym.env): A Gym training environment.
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    return state_size, action_size

def get_model_parameters(model_list):
    """Gather model parameters into a single list.

    Params
    ======
    model_list (List[torch.nn.Module]): A list of PyTorch models."""
    all_parameters = []
    for model in model_list:
        for params in model.parameters():
            all_parameters.append(params)
    return all_parameters

def pkl_dump(data, fname):
    """Dump python object into a pickle file.

    Params
    ======
    data (object): A python object to be persisted to disk.
    fname (str): The path of the output pickle file.
    """
    with open(fname, "wb") as f:
        pkl.dump(data, f)

def yaml_dump(data, fname):
    """Save file in YAML format.

    Params
    ======
    data (dict): The dictionary to be persisted to a YAML file.
    fname (str): The path of the YAML output
    """
    with open(fname, "w") as f:
        yaml.dump(data, f)
