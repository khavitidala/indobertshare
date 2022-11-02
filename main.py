import argparse

from summarization import train, extract, filter as fl

parser = argparse.ArgumentParser() 
parser.add_argument("-m", "--mode", help = "train/extract", default="train")
parser.add_argument("-t", "--task", help = "choose either sum: summarization or par:paraphrase", default="sum")
args = parser.parse_args()

if args.task == "sum":
    if args.mode == "train":
        abs_sum = train.TrainAbsSum()
        abs_sum.train()
    elif args.mode == "extract":
        abs_sum = extract.AbsSumExtract()
        abs_sum.extract()
    elif args.mode == "filter":
        fltr = fl.AbsSumFilter()
        fltr.filter_data()
    else:
        print("Wrong mode!")
elif args.task == "par":
    pass
else:
    print("Wrong task!")