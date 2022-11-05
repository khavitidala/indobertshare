from .base import *

res_dict: dict = {
    "id": [],
    "summary": [],
    "generated_summary": []
}

class AbsSumExtract:
    
    @cached_property
    def tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(extract_model_checkpoint)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        return tokenizer

    @cached_property
    def extract_data(self):
        extract_data = datasets.load_dataset(**extract_data_conf)
        return extract_data
    
    @cached_property
    def model(self):
        model = EncoderDecoderModel.from_pretrained(extract_model_checkpoint)
        model.to("cuda:0")
        return model
    
    def generate_summary(self, batch):
        inputs = self.tokenizer(batch["clean_article"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda:0")
        attention_mask = inputs.attention_mask.to("cuda:0")
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            min_length=15,
            max_length=80, 
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3, # trigram blocking
        )
        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res_dict["id"] += batch["id"]
        res_dict["summary"] += batch["clean_summary"]
        res_dict["generated_summary"] += output_str

        return None
    
    def extract(self):
        name = extract_model_checkpoint.split('/')[-1]
        fname = f"{extract_path}{name}-{extract_data_conf.get('name')}-{extract_data_conf.get('split')}-extraction_results.csv"
        _ = self.extract_data.map(
            self.generate_summary,
            batched=True,
            batch_size=extract_batch_size,
            remove_columns=["clean_article", "id"]
        )
        df = pd.DataFrame(res_dict)
        df.to_csv(fname, index=False)