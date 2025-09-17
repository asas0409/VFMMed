# =================================================================================================
# Usage: nnUNetv2_train [DATASET_NAME_OR_ID] [UNET_CONFIGURATION] [FOLD] --npz
# (add --c for continue training)
# (--val for validation-only (no training))
# =================================================================================================
# DATASET_NAME_OR_ID LIST
# - 1: tumor
# - 2: Lung
# - 3: Synapse
# - 4: ACDC
# =================================================================================================
# UNET_CONFIGURATION LIST
# - UNETR
# - UNETR_FREEZE
# - DINOUNETR
# - DINOUNETR_ENC
# - DINOUNETR_ENC_RELOAD    :    when using the pretrained weight (of frozen encoder). use -pretrained_weights flag with the path to the pretrained weight.
# - BeiTUNETR
# - BeiTUNETR_ENC
# - BeiTUNETR_ENC_RELOAD
# - BLIPUNETR
# - BLIPUNETR_ENC
# - BLIPUNETR_ENC_RELOAD
# - OpenClipUNETR
# - OpenClipUNETR_ENC
# - OpenClipUNETR_ENC_RELOAD
# - SAMUNETR
# - SAMUNETR_ENC
# - SAMUNETR_ENC_RELOAD
# - MEDSAMUNETR
# - MEDSAMUNETR_ENC
# - MEDSAMUNETR_ENC_RELOAD
# =================================================================================================

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 4 DINOUNETR 0 --npz