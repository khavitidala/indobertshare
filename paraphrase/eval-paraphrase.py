# %%
import os, sys
# sys.path.append('../')
# os.chdir('../')

import torch
import shutil
import random
import datasets
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration

from modules.tokenization_indonlg import IndoNLGTokenizer
from utils.train_eval import train, evaluate
from utils.metrics import generation_metrics_fn
from utils.forward_fn import forward_generation
from utils.data_utils import MachineTranslationDataset, GenerationDataLoader

import nltk
nltk.download('punkt')

# %%
###
# common functions
###
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())
    
# Set random seed
set_seed(42)

# %%
tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indobart')
max_seq_len = 128
no_special_token = False
model_type = 'indo-bart'
beam_size = 5

separator_id = 4
speaker_1_id = 5
speaker_2_id = 6

train_batch_size = 8
valid_batch_size = 8
test_batch_size = 32

source_lang = "[indonesian]"
target_lang = "[indonesian]"

src_lid = tokenizer.special_tokens_to_ids[source_lang]
tgt_lid = tokenizer.special_tokens_to_ids[target_lang]

# %%
def load_models():
    bart_model = MBartForConditionalGeneration.from_pretrained('indobenchmark/indobart')
    model = bart_model
    model.config.decoder_start_token_id = tgt_lid
    
    return model

# %%
# Make sure cuda is deterministic
torch.backends.cudnn.deterministic = True

# %%
all_models = [
    # './save/filtered_liputan6-indolem',
    './save/filtered_paracotta',
    # './save/full_paracotta',
    # './save/full_liputan6-merge',
    # './save/full_liputan6-indolem',
    # './save/filtered_liputan6-merge',
    # './save/filtered_merge_all'
] # for demo purpose

# %%
PATH = "/home/bertshare"
MAIN_PATH = PATH+"/paraphrase"

# %%
class ParaphraseDataset(Dataset):
    
    def load_dataset(self, is_expert=True): 
        if is_expert:
            data = datasets.load_dataset(path="indonli", split="test_expert")
        else:
            data = datasets.load_dataset(path="indonli", split="test_lay")
#             ds = []
#             for dsplit in ["train", "validation", "test_lay"]:
#                 ds.append(datasets.load_dataset(path="indonli", split=dsplit))
#             data = datasets.concatenate_datasets(ds) 
        data = data.rename_column("label", "id")
        data = data.rename_column("premise", "text")
        data = data.rename_column("hypothesis", "label")
        return data

    def __init__(self, tokenizer, is_expert=False, *args, **kwargs):
        self.data = self.load_dataset(is_expert)
        self.data = self.data.select(range(32)) # for demo purpose
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        data = self.data[index]
        id, text, label = data['id'], data['text'], data['label']
        input_subwords = self.tokenizer.encode(text.lower(), add_special_tokens=False)
        label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
        return data['id'], input_subwords, input_subwords
    
    def __len__(self):
        return len(self.data)

# %% [markdown]
# ## indoNLI-test_expert

# %%
for saved_models in all_models:
    test_dataset = ParaphraseDataset(tokenizer, is_expert=True, no_special_token=no_special_token, 
                                            speaker_1_id=speaker_1_id, speaker_2_id=speaker_2_id, separator_id=separator_id,
                                            max_token_length=max_seq_len)
    test_loader = GenerationDataLoader(dataset=test_dataset, model_type=model_type, tokenizer=tokenizer, max_seq_len=max_seq_len, 
                                   batch_size=test_batch_size, src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=8, shuffle=False)
    model = load_models()
    model_dir = saved_models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model.load_state_dict(torch.load(model_dir + "/best_model_0.th"))
    device = "cuda0"
    # set a specific cuda device
    if "cuda" in device:
        torch.cuda.set_device(int(device[4:]))
        device = "cuda"
        model = model.cuda()
    test_loss, test_metrics, test_hyp, test_label = evaluate(model, data_loader=test_loader, forward_fn=forward_generation, 
                                                         metrics_fn=generation_metrics_fn, model_type=model_type, 
                                                         tokenizer=tokenizer, beam_size=beam_size, 
                                                         max_seq_len=max_seq_len, is_test=True, 
                                                         device='cuda')
    metrics_scores = []
    result_dfs = []

    metrics_scores.append(test_metrics)
    result_dfs.append(pd.DataFrame({
        'hyp': test_hyp, 
        'label': test_label
    }))

    result_df = pd.concat(result_dfs)
    metric_df = pd.DataFrame.from_records(metrics_scores)

    result_df.to_csv(model_dir + "/indoNLI-test_expert-prediction_result.csv")
    metric_df.describe().to_csv(model_dir + "/indoNLI_expert-test-evaluation_result.csv")