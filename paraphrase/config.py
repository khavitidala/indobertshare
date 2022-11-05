PATH = "/workspace/bertshare"
MAIN_PATH = PATH+"/paraphrase"

# Training Config
batch_size=16
encoder_max_length=128
decoder_max_length=128
valid_num=500
resume_checkpoint=False
# resume_checkpoint=MAIN_PATH+"/checkpoint-100000"

# Filtered ParaCotta
train_data_conf: dict = {
    "path": "filtered_paracotta",
    "name": "canonical",
    "data_dir": MAIN_PATH+"/filtered_paracotta",
    "split": "train"
}
val_data_conf: dict = {
    "path": "id_liputan6",
    "name": "xtreme",
    "data_dir": MAIN_PATH+"summarization/liputan6_data",
    "split": "validation[:25%]"
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
    "output_dir": MAIN_PATH+"/",
    "evaluation_strategy": "steps",
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "predict_with_generate": True,
    "logging_steps": 5000,
    "save_steps": 50000,
    "eval_steps": 20000,
    "warmup_steps": 8000,
    "max_steps": 300000,
    "overwrite_output_dir": True,
    "save_total_limit": 4,
    "fp16": True, 
}

# Extraction Config

extract_batch_size = 32
extract_model_checkpoint = MAIN_PATH+"summarization/checkpoint-150000"
extract_data_conf = {
    "path": "id_liputan6",
    "name": "canonical",
    "data_dir": MAIN_PATH+"summarization/liputan6_data",
    "split": "test"
}
extract_path = MAIN_PATH+"summarization/"

# Filtration Config

FILTER_DATA_FOLDER = MAIN_PATH+"summarization"
filter_num_layer = 9
filter_batch_size = 32

# ParaCotta ID
CODE_NAME = "paracotta_full"
filter_data_from_csv = True
filter_data_path = MAIN_PATH+"summarization/paracotta_full.csv"
col1='references'
col2='paraphrase'

# IndoNLI
# CODE_NAME = "indoNLI"
# filter_data_from_csv = False
# filter_data_path = ""
# filter_data_conf: dict = {
#     "path": "indonli",
#     "split": "train",
# }
# col1 = "premise"
# col2 = "hypothesis"

# Liputan6
# CODE_NAME = "liputan6"
# filter_data_from_csv = True
# filter_data_path = MAIN_PATH+"summarization/liputan6_generated_all.csv"
# col1 = "summary"
# col2 = "generated_summary"

