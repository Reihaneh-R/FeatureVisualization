from torch import nn
from alae.model import Model# ALAE
from torch.nn import functional as Fnc
import torch
class SatelliteALAE(nn.Module):
    def __init__(self,checkpoint_path ):
        super(SatelliteALAE, self).__init__()
        self.layer_count = 7
        self.lod = 6
        self.latent_size = 512
        self.checkpoint_path = checkpoint_path
        self.model =Model(
            startf=64,#cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=self.layer_count,#cfg.MODEL.LAYER_COUNT,
            maxf=512,#cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=512,#cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=0.7,#cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=8,#cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=8,#cfg.MODEL.MAPPING_LAYERS,
            channels=3,#cfg.MODEL.CHANNELS,
            generator="GeneratorDefault",#cfg.MODEL.GENERATOR,
            encoder="EncoderDefault")#cfg.MODEL.ENCODER)

        self.model.requires_grad_(False)
        decoder = self.model.decoder
        encoder = self.model.encoder
        mapping_tl = self.model.mapping_d
        mapping_fl = self.model.mapping_f
        dlatent_avg = self.model.dlatent_avg
        self.model_dict = {
            'discriminator_s': encoder,
            'generator_s': decoder,
            'mapping_tl_s': mapping_tl,
            'mapping_fl_s': mapping_fl,
            'dlatent_avg': dlatent_avg
        }
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
        for name, model_i in self.model_dict.items():
            if name in checkpoint["models"]:
                try:
                    model_dict_c = checkpoint["models"].pop(name)
                    if model_dict_c is not None:
                        self.model_dict[name].load_state_dict(model_dict_c, strict=False)
                    else:
                        print("State dict for model \"%s\" is None " % name)
                except RuntimeError as e:
                    print('%s\nFailed to load: %s\n%s' % ('!' * 160, name, '!' * 160))
                    print('\nFailed to load: %s' % str(e))
            else:
                print("No state dict for model: %s" % name)
        checkpoint.pop('models')

        ######################
    def forward(self,x):
        if 0<=x.min()<=1 and  0<=x.max()<=1:
            x = x * 255
        x = x / 127.5 - 1. # normalization
        styles = torch.zeros(x.shape[0], 1, self.latent_size)
        x = self.model.encoder.from_rgb[self.layer_count - self.lod - 1](x)
        x = Fnc.leaky_relu(x, 0.2)

        device = x.device
        styles = styles.to(device)
        for i in range(self.layer_count - self.lod - 1, self.layer_count):
            x, s1, s2 = self.model.encoder.encode_block[i](x)
            styles[:, 0] += s1 + s2

        latents1 = styles[:, :1].squeeze()
        return latents1