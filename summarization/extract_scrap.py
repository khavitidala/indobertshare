import glob
import torch
import datasets
import statistics
import pandas as pd

from datetime import datetime
from functools import cached_property
from transformers import BertTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel
from config import *


def get_string(sentences):
    all_sentence = []
    for sentence in sentences:
        all_sentence += sentence
    return ' '.join(all_sentence)

def gabung(batch):
    return {
        'id': batch["id"],
        'url': batch["url"],
        'clean_article': [get_string(s) for s in batch['clean_article']],
        'clean_summary': [get_string(s) for s in batch['clean_summary']]
    }

print("Load dataset ...")
files = [f for f in glob.glob("liputan6_extend/*.json")]
dataset = datasets.load_dataset(path="json", data_files=files)

print("Checking liputan6_extended dir ...")
ext = [f for f in glob.glob("liputan6_extended/*.csv")]
print(f"You have extracted {extract_batch_size*len(ext)} data")
print("Continue extraction with the remaining data ...")
dataset = dataset["train"].select(range(extract_batch_size*len(ext), len(files)))

print("Clean dataset ...")
dataset = dataset.map(
    gabung,
    batched=True,
    batch_size=128,
    remove_columns=['id', 'url', 'clean_article', 'clean_summary'],
    num_proc=4
)


class AbsSumExtract:
    
    @cached_property
    def tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(extract_model_checkpoint)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        return tokenizer

    @cached_property
    def extract_data(self):
        return dataset
    
    @cached_property
    def model(self):
        model = EncoderDecoderModel.from_pretrained(extract_model_checkpoint)
        model.to("cuda:0")
        return model
    
    def generate_summary(self, batch):
        res_dict: dict = {
            "id": [],
            "summary": [],
            "generated_summary": []
        }
        inputs = self.tokenizer(batch["clean_article"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda:0")
        attention_mask = inputs.attention_mask.to("cuda:0")
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            min_length=20,
            max_length=80, 
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res_dict["id"] += batch["id"]
        res_dict["summary"] += batch["clean_summary"]
        res_dict["generated_summary"] += output_str
        
        last_id = res_dict["id"][-1]
        fname = f"liputan6_extended/{last_id}-from_scrap-extraction_results.csv"
        
        df = pd.DataFrame(res_dict)
        df.to_csv(fname, index=False)
        
        return None
    
    def extract(self):
        _ = self.extract_data.map(
            self.generate_summary,
            batched=True,
            batch_size=extract_batch_size,
            remove_columns=["clean_article", "id"]
        )

print("Extract summary ...")
extractcls = AbsSumExtract()
extractcls.extract()