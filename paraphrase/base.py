import torch
import datasets
import evaluate
import statistics
import pandas as pd

from datetime import datetime
from functools import cached_property
from indobenchmark import IndoNLGTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from .config import *


class IndoBart:

    @cached_property
    def tokenizer(self): 
        return IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
    
    @cached_property
    def bertscore(self):
        return evaluate.load("bertscore")
    
    @cached_property
    def sacrebleu(self):
        return evaluate.load("sacrebleu")
    
    @cached_property
    def indobart(self):
        if from_checkpoint:
            bart_model = MBartForConditionalGeneration.from_pretrained(from_checkpoint)
            bart_model.to("cuda:0")
            return bart_model

        bart_model = MBartForConditionalGeneration.from_pretrained('indobenchmark/indobart-v2')
        for k, v in indobart_conf.items():
            setattr(bart_model.config, k, v)

        return bart_model

    def process_data_to_model_inputs(self, batch):
        inputs = self.tokenizer(batch["clean_article"], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = self.tokenizer(batch["clean_summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        df = pd.DataFrame({"references": pred_str,
                           "paraphrase": label_str})
        df.to_csv(f"{MAIN_PATH}/output_logs/{str(int(datetime.now().timestamp()))}.csv", index=False)
        
        bert_score = self.bertscore.compute(
                        predictions=pred_str,
                        references=label_str,
                        verbose=True,
                        device="cuda:0",
                        lang="id",
                        model_type="bert-base-multilingual-cased",
                        num_layers=9,
                        use_fast_tokenizer=False
                    )
        ibleu = self.sacrebleu.compute(predictions=pred_str, references=label_str)
        
        return {
            "bert_score": round(statistics.mean(bert_score["f1"])*100, 4),
            "inverse_bleu": round(100 - ibleu["score"], 4)
        }