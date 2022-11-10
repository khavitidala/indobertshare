from .base import *


class TrainIndoBart(IndoBart):
    
    @cached_property
    def all_data(self):
        return datasets.load_dataset(split="train", **data_conf)
    
    @cached_property
    def train_data(self):
        train_ds = self.all_data.select(range(0, self.all_data.num_rows-100))
        if train_num:
            train_ds = train_ds.select(range(train_num))
        train_ds = train_ds.map(
            self.process_data_to_model_inputs, 
            batched=True, 
            batch_size=batch_size, 
            remove_columns=[col1, col2],
            num_proc=4
        )
        train_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
        return train_ds
    
    @cached_property
    def val_data(self):
        val_ds = self.all_data.select(range(self.all_data.num_rows-100, self.all_data.num_rows))
        if valid_num:
            val_ds = val_ds.select(range(valid_num))
        val_ds = val_ds.map(
            self.process_data_to_model_inputs, 
            batched=True, 
            batch_size=batch_size, 
            remove_columns=[col1, col2],
            num_proc=4
        )
        val_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
        return val_ds
    
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
