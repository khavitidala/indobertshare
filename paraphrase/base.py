import torch
import datasets
import evaluate
import statistics
import pandas as pd

from copy import deepcopy
from datetime import datetime
from functools import cached_property
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, MBartForConditionalGeneration
from .config import *

from .tokenization_indonlg import IndoNLGTokenizer



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
        bart_model.config.decoder_start_token_id = self.tokenizer.special_tokens_to_ids["[indonesian]"]
        for k, v in indobart_conf.items():
            setattr(bart_model.config, k, v)

        return bart_model

    def process_data_to_model_inputs(self, batch):
        self.tokenizer.truncation=True
        self.tokenizer.max_length=encoder_max_length
        results = self.tokenizer.prepare_input_for_generation(
            inputs=batch[col1],
            decoder_inputs=batch[col2],
            padding="max_length"
        )
        
        for k, v in results.items():
            batch[k] = v

        batch["labels"] = deepcopy(results["decoder_input_ids"])
        batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        df = pd.DataFrame({"references": label_str,
                           "paraphrase": pred_str})
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