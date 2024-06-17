import torch
import torch.nn as nn


class VAE_Transformer(nn.Module):
    def __init__(self, vae_model, transformer_model):
        super(VAE_Transformer, self).__init__()
        self.vae_model = vae_model
        self.transformer_model = transformer_model

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # print("IN VAR: ", batch_x.shape)
        out_var = self.vae_model(batch_x)
        # print("OUT VAR: ", out_var[1].shape)
        in_transformer = torch.cat((batch_x, out_var[1]), axis=-1)
        # print("IN XMER: ", in_transformer.shape)
        out_transformer = self.transformer_model(in_transformer, batch_x_mark, dec_inp, batch_y_mark)
        # print("OUT XMER: ", out_transformer.shape)
        return out_var, out_transformer
