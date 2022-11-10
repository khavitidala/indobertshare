PATH = "/workspace/bertshare"
MAIN_PATH = PATH+"/paraphrase"

# Training Config
batch_size=16
encoder_max_length=128
decoder_max_length=128
train_num=False
valid_num=100
from_checkpoint=False
resume_checkpoint=False
# resume_checkpoint=MAIN_PATH+"/checkpoint-100000"

# col1='references'
# col2='paraphrase'

# # Filtered ParaCotta
# data_conf: dict = {
#     "path": "filtered_paracotta",
#     "data_dir": MAIN_PATH+"/filtered_paracotta"
# }

# # Full ParaCotta
# data_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/full_paracotta.csv"
# }

col1 = "summary"
col2 = "generated_summary"

# # filtered_liputan6-indonlu
# data_conf: dict = {
#     "path": "filtered_liputan6-indonlu"
# }

# # full_liputan6-indonlu
# data_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/full_liputan6-indonlu.csv"
# }

# # full_liputan6-indolem
# data_conf: dict = {
#     "path": "csv",
#     "data_dir": MAIN_PATH+"/data",
#     "data_files": MAIN_PATH+"/data/full_liputan6-indolem.csv"
# }

OUTPUT_DIR = MAIN_PATH+"/filtered_liputan6-indolem/"
data_conf: dict = {
    "path": "csv",
    "data_dir": MAIN_PATH+"/data",
    "data_files": MAIN_PATH+"/data/filtered_liputan6-indolem.csv"
}

indobart_conf: dict = {
    "max_length": 80,
    "min_length": 15,
    "no_repeat_ngram_size": 3, # trigram blocking
    "early_stopping":  True,
    "length_penalty": 2.0,
    "num_beams": 5
}
seq2seq_args: dict = {
    "output_dir": OUTPUT_DIR,
    "evaluation_strategy": "steps",
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "predict_with_generate": True,
    "logging_steps": 5000,
    "save_steps": 25000,
    "eval_steps": 25000,
    "warmup_steps": 8000,
    "max_steps": 200000,
    "overwrite_output_dir": True,
    "save_total_limit": 4,
    "fp16": True, 
}

