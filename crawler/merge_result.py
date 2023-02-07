import glob
import argparse
import pandas as pd

parser = argparse.ArgumentParser() 
parser.add_argument("-n", "--name", help = "name of output", default="urls")
parser.add_argument("-t", "--file_format", help = "file format", default="json")
parser.add_argument("-f", "--folder_name", help = "folder name of all json files", default="results")
args = parser.parse_args()

result = []
for f in glob.glob(f"{args.folder_name}/*.{args.file_format}"):
    with open(f, "r") as infile:
        if args.file_format == "json":
            result.append(pd.read_json(infile))
        elif args.file_format == "csv":
            result.append(pd.read_csv(infile))
        else:
            print(f"The {args.file_format} format is not supported ")
            raise Exception
mergejson = pd.concat(result)
mergejson.to_csv(f"{args.name}.csv", index=False, header=False)