import torch
import torch.nn as nn

from nnunetv2.custom_model.encoder import MedSam2dEncoder
from nnunetv2.custom_model.decoder_3D import SAMDecoder3D

import numpy as np



class deepSupervisionDummy():
    def __init__(self):
        self.deep_supervision = False
    
class MedSAMUNETR(nn.Module) :
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(MedSAMUNETR, self).__init__()

        self.decoder = deepSupervisionDummy()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = MedSam2dEncoder(freeze=True)
        self.decoder1 = SAMDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim, ds = self.decoder.deep_supervision)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        B, C, D, H, W= x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(B * D, C, H, W).contiguous()
        
        t1   = x[:, 0:1, ...].repeat(1, 3, 1, 1)
        t2   = x[:, 1:2, ...].repeat(1, 3, 1, 1)
        t1ce = x[:, 2:3, ...].repeat(1, 3, 1, 1)
        flair= x[:, 3:4, ...].repeat(1, 3, 1, 1)
        
        modalities = [t1,t2,t1ce,flair]
        
        slot = []
        for i in range(C):
            one_modality = modalities[i]
            features = self.encoder(one_modality)
               
            reshaped_features = [
                f.view(B, D, f.shape[1], f.shape[2], f.shape[3]).contiguous() for f in features
            ]
            slot.append(reshaped_features)
        
        
        if C == 4:    
            encoder_features = [torch.cat([slot[0][i], slot[1][i], slot[2][i], slot[3][i]], dim=4) for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0] 
        
        output = self.decoder1(x.view(B, D, C, H, W).contiguous(), encoder_features)
        
        return output
    
class MedSAMUNETR_ENC(nn.Module) :
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(MedSAMUNETR_ENC, self).__init__()

        self.decoder = deepSupervisionDummy()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = MedSam2dEncoder(freeze=False)
        self.decoder1 = SAMDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim, ds = self.decoder.deep_supervision)

    def forward(self, x):

        B, C, D, H, W= x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(B * D, C, H, W).contiguous()
        
        t1   = x[:, 0:1, ...].repeat(1, 3, 1, 1)
        t2   = x[:, 1:2, ...].repeat(1, 3, 1, 1)
        t1ce = x[:, 2:3, ...].repeat(1, 3, 1, 1)
        flair= x[:, 3:4, ...].repeat(1, 3, 1, 1)
        
        modalities = [t1,t2,t1ce,flair]
        
        slot = []
        for i in range(C):
            one_modality = modalities[i]
            features = self.encoder(one_modality)
               
            reshaped_features = [
                f.view(B, D, f.shape[1], f.shape[2], f.shape[3]).contiguous() for f in features
            ]
            slot.append(reshaped_features)
        
        
        if C == 4:    
            encoder_features = [torch.cat([slot[0][i], slot[1][i], slot[2][i], slot[3][i]], dim=4) for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0] 
        
        output = self.decoder1(x.view(B, D, C, H, W).contiguous(), encoder_features)
        
        return output


#test

if __name__ == '__main__':
    model = MedSAMUNETR(input_dim=4, output_dim=4)
    # x = torch.randn( 1, 4, 3, 224,224 ) #B, D, C, H, W
    x = torch.randn(1, 4, 32, 224,224 ) #B, C, D, H, W
    output = model(x)
    
    print(f"UNETR output_shape : {output.shape}")
