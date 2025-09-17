# =================================================================================================
# Usage: nnUNetv2_train [DATASET_NAME_OR_ID] [UNET_CONFIGURATION] [FOLD] --npz
# (add --c for continue training)
# =================================================================================================
# UNET_CONFIGURATION LIST
# - UNETR
# - DINOUNETR
# - DINOUNETR_ENC
# - BeiTUNETR
# - BeiTUNETR_ENC
# - BLIPUNETR
# - BLIPUNETR_ENC
# - OpenClipUNETR
# - OpenClipUNETR_ENC
# - SAMUNETR
# - SAMUNETR_ENC
# - MEDSAMUNETR
# - MEDSAMUNETR_ENC
# =================================================================================================

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 DINOUNETR 1 --npz &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 1 DINOUNETR 2 --npz &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 1 DINOUNETR 3 --npz 
#wait
