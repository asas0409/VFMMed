import torch.nn as nn
import torch

from nnunetv2.custom_model.encoder import BLIPEncoder
from nnunetv2.custom_model.decoder_3D import UNETRDecoder3D

class deepSupervisionDummy():
    def __init__(self):
        self.deep_supervision = False


class BLIPUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(BLIPUNETR, self).__init__()
        
        self.decoder = deepSupervisionDummy()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = BLIPEncoder(freeze=True)
        self.decoder1 = UNETRDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim, ds = self.decoder.deep_supervision) 

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
                # 2, 4, 16, 224, 224
        B, C, D, H, W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        x= x.view(B * D, C, H, W).contiguous()
        # x : 32, 4, 224, 224
        
        # t1 : 128, 3, 224, 224
        t1   = x[:, 0:1, ...].repeat(1, 3, 1, 1)
        t2   = x[:, 1:2, ...].repeat(1, 3, 1, 1)
        t1ce = x[:, 2:3, ...].repeat(1, 3, 1, 1)
        flair= x[:, 3:4, ...].repeat(1, 3, 1, 1)

        modalities = [t1, t2, t1ce, flair]

        slot = []
        for i in range(C):
            one_modality = modalities[i]
            features = self.encoder(one_modality)
            # feature : 32, 256, 768
            reshaped_features = [
                f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
            ]
            # feature : 2, 16, 256, 768

            slot.append(reshaped_features)
        
        if C == 4:
            encoder_features = [torch.cat([slot[0][i], slot[1][i], slot[2][i], slot[3][i]], dim=3) for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0]
        
        output = self.decoder1(x.view(B, D, C, H, W).contiguous(), encoder_features)
        return output
    
class BLIPUNETR_ENC(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(BLIPUNETR_ENC, self).__init__()
        
        self.decoder = deepSupervisionDummy()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = BLIPEncoder(freeze=False)
        self.decoder1 = UNETRDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim, ds = self.decoder.deep_supervision) 

            
    def forward(self, x):
                # 2, 4, 16, 224, 224
        B, C, D, H, W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        x= x.view(B * D, C, H, W).contiguous()
        # x : 32, 4, 224, 224
        
        # t1 : 128, 3, 224, 224
        t1   = x[:, 0:1, ...].repeat(1, 3, 1, 1)
        t2   = x[:, 1:2, ...].repeat(1, 3, 1, 1)
        t1ce = x[:, 2:3, ...].repeat(1, 3, 1, 1)
        flair= x[:, 3:4, ...].repeat(1, 3, 1, 1)

        modalities = [t1, t2, t1ce, flair]

        slot = []
        for i in range(C):
            one_modality = modalities[i]
            features = self.encoder(one_modality)
            # feature : 32, 256, 768
            reshaped_features = [
                f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
            ]
            # feature : 2, 16, 256, 768

            slot.append(reshaped_features)
        
        if C == 4:
            encoder_features = [torch.cat([slot[0][i], slot[1][i], slot[2][i], slot[3][i]], dim=3) for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0]
        
        output = self.decoder1(x.view(B, D, C, H, W).contiguous(), encoder_features)
        return output


if __name__ == '__main__':
    model = BLIPUNETR()
    # x = torch.randn( 1, 4, 3, 1024,1024 ) #B, D, C, H, W
    x = torch.randn( 2, 4, 64, 224,224 ) #B, C, D, H, W
    output = model(x)
    
    print(f"UNETR output_shape : {output.shape}")