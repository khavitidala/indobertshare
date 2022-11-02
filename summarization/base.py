import torch
import datasets
import statistics
import pandas as pd

from datetime import datetime
from functools import cached_property
from transformers import BertTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel
from .config import *


class AbsSum:

    @cached_property
    def tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        return tokenizer

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
    
    @cached_property
    def bert2bert(self):
        bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_model, bert_model, tie_encoder_decoder=tie_encoder_decoder)
        bert2bert.config.decoder_start_token_id = self.tokenizer.bos_token_id
        bert2bert.config.eos_token_id = self.tokenizer.eos_token_id
        bert2bert.config.pad_token_id = self.tokenizer.pad_token_id
        bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
        
        for k, v in bert2bert_conf.items():
            setattr(bert2bert.config, k, v)
        
        return bert2bert

    @cached_property
    def rouge(self):
        return datasets.load_metric("rouge")

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        with open("compute_metrics_output.txt", "w") as f:
            f.write(str(pred_str)+"\n"+str(label_str))
        
        rouge_output = self.rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }