from .base import *

import evaluate


class AbsSumFilter:
    def __init__(self):
        self.col1 = col1
        self.col2 = col2
        self.res = {
                col1: [],
                col2: [],
                "bert_score": []
            }

    @cached_property
    def extract_data(self):
        if CODE_NAME == "indoNLI":
            if not is_expert:
                ds = []
                for dsplit in ["train", "validation", "test_lay"]:
                    filter_data_conf["split"] = dsplit
                    ds.append(datasets.load_dataset(**filter_data_conf))
                extract_data = datasets.concatenate_datasets(ds)
            else:
                filter_data_conf["split"] = "test_expert"
                extract_data = datasets.load_dataset(**filter_data_conf)
            extract_data = extract_data.filter(lambda x: x["label"] != 2) # 2 means contradictions
            extract_data = extract_data.remove_columns(["label"])
            return extract_data
        
        if filter_data_from_csv:
            df = pd.read_csv(filter_data_path)
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            extract_data = datasets.Dataset.from_pandas(df)
        else:
            extract_data = datasets.load_dataset(**filter_data_conf)
            if not filter_data_conf.__contains__("split"):
                extract_data = extract_data["train"]
        # extract_data = extract_data.select(range(filter_batch_size))
        return extract_data

    @cached_property
    def bertscore(self):
        return evaluate.load("bertscore")
    
    @cached_property
    def sacrebleu(self):
        return evaluate.load("sacrebleu")

    def calculate_bertscore(self, batch):
        try:
            results = self.bertscore.compute(
                predictions=batch[self.col1],
                references=batch[self.col2],
                verbose=True,
                device="cuda:0",
                lang="id",
                model_type="bert-base-multilingual-cased",
                num_layers=9,
                use_fast_tokenizer=False
            )
        except Exception as e:
            with open(f"log-{str(int(datetime.now().timestamp()))}.txt", "w") as f:
                f.write(str(e))
            return None
        # results = {}
        # results["f1"] = [0.1 for _ in range(len(batch[self.col1]))]
        self.res[self.col1] += batch[self.col1]
        self.res[self.col2] += batch[self.col2]
        self.res["bert_score"] += [round(r*100, 2) for r in results["f1"]]

        return None
    
    def calculate_ibleu(self, text):
        try:
            res = self.sacrebleu.compute(predictions=[text[self.col1]], references=[text[self.col2]])
        except Exception as e:
            with open(f"log-{str(int(datetime.now().timestamp()))}.txt", "w") as f:
                f.write(str(e))
            self.res["ibleu_score"].append(0)
            return None
        
        self.res["ibleu_score"].append(round(100 - res["score"], 2))
        
        return None

    def filter_data(self):
        torch.cuda.empty_cache()
        print("phase I.1: calculate BERTScore for all data")
        _ = self.extract_data.map(
            self.calculate_bertscore,
            batched=True,
            batch_size=filter_batch_size,
            remove_columns=[self.col1, self.col2],
        )
        # fname = f"{MAIN_PATH}paraphrase/{CODE_NAME}-{str(int(datetime.now().timestamp()))}"
        fname = f"{MAIN_PATH}paraphrase/{CODE_NAME}"
        if is_expert:
            fname = fname+"-expert"
        dfs = pd.DataFrame(self.res)
        dfs.to_csv(f"{fname}-phase1.csv", index=False)

        print("phase I.2: get average BERTScore from all data")
        avg_bertscore = statistics.mean(self.res["bert_score"])
        print("average BERTScore: "+str(avg_bertscore))
        with open(f"{fname}-avg_bertscore.txt", "w") as f:
            f.write(str(avg_bertscore))
        
        print("phase I.3: filter data by score > average BERTScore")
        self.extract_data = datasets.Dataset.from_pandas(dfs)
        self.extract_data = self.extract_data.filter(lambda x: x["bert_score"] > avg_bertscore)

        print("phase II.1: calculate inverse BLEU for all data")
        self.res = {
                        col1: self.extract_data[col1],
                        col2: self.extract_data[col2],
                        "bert_score": self.extract_data["bert_score"],
                        "ibleu_score": []
                    }
        _ = self.extract_data.map(self.calculate_ibleu, batched=False, remove_columns=[self.col1, self.col2])
        try:
            dfs = pd.DataFrame(self.res)
            dfs.to_csv(f"{fname}-phase2.csv", index=False)

            print("phase II.2: get average inverse BLEU from all data")
            avg_ibleu = statistics.mean(self.res["ibleu_score"])
            print("average inverse BLEU: "+str(avg_ibleu))
            with open(f"{fname}-avg_ibleu.txt", "w") as f:
                f.write(str(avg_ibleu))

            print("phase III.3: filter data by score > average inverse BLEU")
            self.extract_data = datasets.Dataset.from_pandas(dfs)
            self.extract_data = self.extract_data.filter(lambda x: x["ibleu_score"] > avg_ibleu)

            print("save the final result")
            # self.extract_data.save_to_disk(fname)
            self.extract_data.to_csv(f"{fname}-final.csv")
            self.extract_data.to_csv(f"{MAIN_PATH}paraphrase/data/{CODE_NAME}.csv") # for demo purpose
        
        except Exception:
            pass