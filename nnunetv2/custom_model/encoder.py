import torch
import numpy as np
import cv2

import torch.nn as nn
from transformers import AutoModel, BeitModel, BlipModel, CLIPModel, SamModel
from albumentations.pytorch import ToTensorV2
import albumentations as A
from argparse import Namespace
from nnunetv2.custom_model.MedSAM.Med2D.segment_anything import medsam_model_registry
from nnunetv2.custom_model.SAM.segment_anything import sam_model_registry



class Dinov2Encoder(nn.Module):
    def __init__(self, freeze = True, model_name = 'facebook/dinov2-base'):
        super(Dinov2Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze

    def forward(self, x):
        outputs = []
        if self.freeze:
            with torch.no_grad(): 
                x = self.model.embeddings(x) #patch로 나누고 embedding (positional encoding도 하도록 수정)
                for i, layer in enumerate(self.model.encoder.layer):
                    x = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
                    if i+1 in [3, 6, 9, 12]:
                        outputs.append(x)
        else:
            x = self.model.embeddings(x) #patch로 나누고 embedding (positional encoding도 하도록 수정)
            for i, layer in enumerate(self.model.encoder.layer):
                x = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
                if i+1 in [3, 6, 9, 12]:
                    outputs.append(x)
                    
        return outputs

class BEiTv2Encoder(nn.Module):
    def __init__(self, freeze = True, model_name='microsoft/beit-base-patch16-224-pt22k-ft22k'):
        super(BEiTv2Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze

    
    def forward(self, x):
        outputs = []
        if self.freeze:
            with torch.no_grad():
                batch_size, channels, height, width = x.size()
                resolution = (height, width)
                embedding_output = self.model.embeddings(x)[0]
                
                for i, layer in enumerate(self.model.encoder.layer):
                    # 각 레이어 호출
                    layer_outputs = layer(embedding_output, resolution=resolution)
                    
                    if isinstance(layer_outputs, tuple):
                        embedding_output = layer_outputs[0]
                    else : 
                        embedding_output = layer_outputs
                    if i + 1 in [3, 6, 9, 12]:
                        outputs.append(embedding_output)
        else:
            batch_size, channels, height, width = x.size()
            resolution = (height, width)
            embedding_output = self.model.embeddings(x)[0]
            
            for i, layer in enumerate(self.model.encoder.layer):
                # 각 레이어 호출
                layer_outputs = layer(embedding_output, resolution=resolution)
                
                if isinstance(layer_outputs, tuple):
                    embedding_output = layer_outputs[0]
                else : 
                    embedding_output = layer_outputs
                if i + 1 in [3, 6, 9, 12]:
                    outputs.append(embedding_output)
        
        return outputs
        

class BLIPEncoder(nn.Module):
    def __init__(self, freeze = True, model_name='Salesforce/blip-image-captioning-base'):
        """
        BLIP Vision Model 초기화
        """
        super(BLIPEncoder, self).__init__()
        self.model = BlipModel.from_pretrained(model_name).vision_model
        self.freeze = freeze
        self.encoder_layers = self.model.encoder.layers
        self.embeddings = self.model.embeddings
        self.target_layers = [2, 5, 8, 11]  # 0-index 기준

    def forward(self, x):
        outputs = []  # 선택된 레이어 출력을 저장할 리스트
        if self.freeze:
            with torch.no_grad():
                # 이미지 입력을 Embedding 단계에 전달
                embeddings = self.embeddings(x)
                # embeddings = self.dropout(embeddings)
                
                # Attention Mask 생성
                batch_size, seq_len, _ = embeddings.size()
                #print(f"batch_size, seq_len, _ : {batch_size, seq_len, _}")
                attention_mask = torch.ones(batch_size, 1, 1, seq_len).to(embeddings.device)

                
                # Vision Transformer Encoder 레이어를 순차적으로 통과
                for i, layer in enumerate(self.encoder_layers):
                    outputs_layer = layer(embeddings, attention_mask=attention_mask)
                    embeddings = outputs_layer[0] if isinstance(outputs_layer, tuple) else outputs_layer
                    
                    # 대상 레이어 출력 저장
                    if i in self.target_layers:
                        outputs.append(embeddings)
        else:
            # 이미지 입력을 Embedding 단계에 전달
            embeddings = self.embeddings(x)
            # embeddings = self.dropout(embeddings)
            
            # Attention Mask 생성
            batch_size, seq_len, _ = embeddings.size()
            #print(f"batch_size, seq_len, _ : {batch_size, seq_len, _}")
            attention_mask = torch.ones(batch_size, 1, 1, seq_len).to(embeddings.device)

            
            # Vision Transformer Encoder 레이어를 순차적으로 통과
            for i, layer in enumerate(self.encoder_layers):
                outputs_layer = layer(embeddings, attention_mask=attention_mask)
                embeddings = outputs_layer[0] if isinstance(outputs_layer, tuple) else outputs_layer
                
                # 대상 레이어 출력 저장
                if i in self.target_layers:
                    outputs.append(embeddings)
        
        return outputs


class OpenClipEncoder(nn.Module):
    def __init__(self, freeze=True, model_name='openai/clip-vit-base-patch16'):
       super(OpenClipEncoder, self).__init__()
       self.model = CLIPModel.from_pretrained(model_name)
       self.freeze = freeze
       self.vision_model = self.model.vision_model
       self.encoder_layers = self.vision_model.encoder.layers
       
    def forward(self, x) :
        selected_outputs = []
        if self.freeze:
            with torch.no_grad():
                embeddings = self.vision_model.embeddings(x)
                
                batch_size, seq_len, _ = embeddings.size()
                attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(embeddings.device)
                causal_attention_mask = torch.zeros(batch_size,1, seq_len, seq_len).to(embeddings.device)
                
                for i, layer in enumerate(self.encoder_layers):
                    outputs = layer(embeddings, attention_mask = attention_mask, causal_attention_mask=causal_attention_mask)
                    embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                    if i+1 in [3, 6, 9, 12]:
                        selected_outputs.append(embeddings)
        else:
            embeddings = self.vision_model.embeddings(x)
                
            batch_size, seq_len, _ = embeddings.size()
            attention_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(embeddings.device)
            causal_attention_mask = torch.zeros(batch_size,1, seq_len, seq_len).to(embeddings.device)
            
            for i, layer in enumerate(self.encoder_layers):
                outputs = layer(embeddings, attention_mask = attention_mask, causal_attention_mask=causal_attention_mask)
                embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
                if i+1 in [3, 6, 9, 12]:
                    selected_outputs.append(embeddings)
                    
        return selected_outputs    
    

class SamEncoder(nn.Module):
    def __init__(self, out_channels, freeze = True):
        super(SamEncoder, self).__init__()
        self.model = sam_model_registry["vit_b"](image_size=224,
                                                num_classes=out_channels,
                                                checkpoint='/app/nnUNet/nnunetv2/custom_model/SAM/pretrained_weight/sam_vit_b_01ec64.pth',
                                                in_channel=3,
                                                pixel_mean=[0, 0, 0],
                                                pixel_std=[1, 1, 1]).image_encoder
        self.freeze = freeze
        
    def forward(self, x):
        outputs = []
        b, c, h, w = x.size()
        
        if self.freeze:
            with torch.no_grad():
                # Patch Embedding
                x = self.model.patch_embed(x)
                if self.model.pos_embed is not None:
                    x = x + self.model.pos_embed
                    
                # Vision Transformer Layers
                for i, blk in enumerate(self.model.blocks):
                    x = blk(x)
                    if i + 1 in [3, 6, 9, 12]:
                        outputs.append(x) #[B*D, 64, 64, 768]
        else:
            # Patch Embedding
            x = self.model.patch_embed(x)
            if self.model.pos_embed is not None:
                x = x + self.model.pos_embed
                
            # Vision Transformer Layers
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
                if i + 1 in [3, 6, 9, 12]:
                    outputs.append(x) #[B*D, 64, 64, 768]
                
        return outputs 

args = Namespace()
args.image_size = 224
args.sam_checkpoint = '/app/nnUNet/nnunetv2/custom_model/MedSAM/pretrained_weight/sam-med2d_b.pth'
args.encoder_adapter = True

class MedSam2dEncoder(nn.Module):
    def __init__(self, freeze = True):
        super(MedSam2dEncoder, self).__init__()
        self.model = medsam_model_registry["vit_b"](args).image_encoder
        self.freeze = freeze
        
    def forward(self, x):
        outputs = []
        if self.freeze:
            with torch.no_grad():
                x = self.model.patch_embed(x)
                if self.model.pos_embed is not None:
                    x = x + self.model.pos_embed    
                # Vision Transformer Layers
                for i, blk in enumerate(self.model.blocks):
                    x = blk(x)
                    if i + 1 in [3, 6, 9, 12]:
                        outputs.append(x) #[B*D, 64, 64, 768]
        else:
            x = self.model.patch_embed(x)
            if self.model.pos_embed is not None:
                x = x + self.model.pos_embed
            # Vision Transformer Layers
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
                if i + 1 in [3, 6, 9, 12]:
                    outputs.append(x) #[B*D, 64, 64, 768]
            
        return outputs

        
        
        

if __name__ == '__main__':

    dummy_image = torch.randn(1, 3, 1024, 1024)  # (batch_size, channels, height, width)
    encoder = SamEncoder()
    features = encoder(dummy_image)

    print(len(features))
    print(features[0].shape)
