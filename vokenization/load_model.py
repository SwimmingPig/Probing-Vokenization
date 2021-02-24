# import torch
from transformers import AutoConfig, TFAutoModel, FeatureExtractionPipeline
from transformers import BertConfig, BertModel, TFBertModel

PATH = './vlm_12L_768H_wiki/'

# config = AutoConfig.from_pretrained(PATH + 'config.json')
# model = TFAutoModel.from_pretrained(PATH + 'pytorch_model.bin', from_pt=True, config=config)


config = BertConfig.from_json_file(PATH + 'config.json')
model = BertModel.from_pretrained(PATH + 'pytorch_model.bin', from_tf=False, config=config)
# model = TFBertModel.from_pretrained(PATH + 'pytorch_model.bin', from_pt=True, config=config)
# model.eval()
# model.to(device)
print(model.config)

# print(model.get_all_encoder_layers())