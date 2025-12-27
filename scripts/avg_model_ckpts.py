#!/usr/bin/env python

import sys
import torch
import torch.nn as nn
from collections import OrderedDict

torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)


sys.path.append("~/src/jenever/src/dnaseq2seq")

from dnaseq2seq.model import VarTransformer


def load_model(model_path):
    """
    Create the VariantTransformer model using params / config settings from the given path
    Model is compiled with torch.compile and set to eval mode
    :returns: VariantTransformer model with parameters loaded
    """
    model_info = torch.load(model_path, map_location='cpu', weights_only=False)
    statedict = model_info['model_state_dict']
    modelconf = model_info['conf']
    new_state_dict = {}
    for key in statedict.keys():
      new_key = key.replace('_orig_mod.', '')
      new_state_dict[new_key] = statedict[key]
    statedict = new_state_dict

    model = VarTransformer(read_depth=modelconf['max_read_depth'],
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=260,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           decoder_embed_dim=modelconf['decoder_embed_dim'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           d_ff=modelconf['dim_feedforward'],
                           device='cpu')

    model.load_state_dict(statedict, strict=False)


    return model, modelconf


def average_model_parameters(model_list, modelconf):
    """
    Averages the parameters of the given list of models.
    All models must have the same architecture.

      :param model_list: List of models (instances of nn.Module)
      :return: A new model instance (nn.Module) with averaged parameters
    """
    # Start by getting a copy of the first model's state_dict
    # which will be used to store the average parameters
    average_state_dict = OrderedDict(model_list[0].state_dict())
    for key in average_state_dict:
        if key == "decoder0.layers.3.linear1.bias":
            print(f"{key}: {average_state_dict[key][0:20]}")

    # Initialize the averaged state dict with zeros
    #for key in average_state_dict:
    #    average_state_dict[key].zero_()

    # Sum all model parameters
    for model in model_list[1:]:
        model_state_dict = model.state_dict()
        for key in average_state_dict:
            if key == "decoder0.layers.3.linear1.bias":
                print(f"{key}: {model_state_dict[key][0:20]}")
            average_state_dict[key] += model_state_dict[key]

    # Divide by the number of models to get the average
    print(f"Final avg params:")
    for key in average_state_dict:
        average_state_dict[key] /= len(model_list)
        if key == "decoder0.layers.3.linear1.bias":
            print(f"{key}: {average_state_dict[key][0:20]}")


    new_model = VarTransformer(read_depth=modelconf['max_read_depth'],
                               feature_count=modelconf['feats_per_read'],
                               kmer_dim=260,  # Number of possible kmers
                               n_encoder_layers=modelconf['encoder_layers'],
                               n_decoder_layers=modelconf['decoder_layers'],
                               embed_dim_factor=modelconf['embed_dim_factor'],
                               decoder_embed_dim=modelconf['decoder_embed_dim'],
                               encoder_attention_heads=modelconf['encoder_attention_heads'],
                               decoder_attention_heads=modelconf['decoder_attention_heads'],
                               d_ff=modelconf['dim_feedforward'],
                               device='cpu')
    new_model.load_state_dict(average_state_dict, strict=False)
    return new_model


def main(paths):
    models = []
    modelconf = None
    for path in paths:
        print(f"Loading model from {path}")
        model, modelconf = load_model(path)
        models.append(model)
    avg_model = average_model_parameters(models, modelconf)

    final = {
        "model_state_dict": avg_model.half().state_dict(),
        "conf": modelconf,

        "model_average_from": [paths]
    }
    torch.save(final, "averaged_model.pt")

if __name__=="__main__":
    main(sys.argv[1:])

