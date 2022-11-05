from .base import *


class TrainIndoBart(IndoBart):
    
    @cached_property
    def train_data(self):
        train_data = datasets.load_dataset(**train_data_conf)
        if train_num:
            train_data = train_data.select(range(train_num))
        train_data = train_data.map(
            self.process_data_to_model_inputs, 
            batched=True, 
            batch_size=batch_size, 
            remove_columns=['id', 'url', 'clean_article', 'clean_summary', 'extractive_summary'],
            num_proc=4
        )
        train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
        return train_data
    
    @cached_property
    def val_data(self):
        val_data = datasets.load_dataset(**val_data_conf)
        if valid_num:
            val_data = val_data.select(range(valid_num))
        val_data = val_data.map(
            self.process_data_to_model_inputs, 
            batched=True, 
            batch_size=batch_size, 
            remove_columns=['id', 'url', 'clean_article', 'clean_summary', 'extractive_summary'],
            num_proc=4
        )
        val_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
        return val_data
    
    def train(self):
        training_args = Seq2SeqTrainingArguments(**seq2seq_args)
        trainer = Seq2SeqTrainer(
            model=self.indobart,
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
        )
        if resume_checkpoint:
            trainer.train(model_path=resume_checkpoint)
        else:
            trainer.train()
