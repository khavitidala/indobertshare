import json
import time
import argparse
import numpy as np
import pandas as pd
from async_scrape import AsyncScrape
from bs4 import BeautifulSoup


parser = argparse.ArgumentParser() 
parser.add_argument("-t", "--target_path", help = "path folder where all results will be save", default="results")
parser.add_argument("-b", "--batch", help = "how many url(s) will be proceed every iteration", default=1000)
parser.add_argument("-r", "--rest_time", help = "how long time.sleep call for every iteration", default=5)
# parser.add_argument("-e", "--epoch", help = "number of iteration", default=5)
parser.add_argument("-f", "--url_path", help = "url path folder containing the csv file", default="url-0.csv")
args = parser.parse_args()


def get_id(url):
    id = [s for s in url.split("/") if s.isdigit()]
    return id[0]

def get_summary(text):
    target = ''
    for line in text.split('\n'):
        if 'window.kmklabs.channel =' in line:
            target = line
            break
    temp=target.split('window.kmklabs.article = ')[1]
    temp=temp.split(';')[0]
    data = json.loads(temp)
    return data['shortDescription']

def extract_data(text):
    soup = BeautifulSoup(text, 'lxml')
    title = soup.findAll('title')[0].getText().replace(' - News Liputan6.com', '')
    date = soup.findAll('time', {'class': 'read-page--header--author__datetime updated'})[0].getText()
    article = []
    contents = soup.findAll('div', {'class': 'article-content-body__item-content'})
    for content in contents:
        article.append(content.getText())
    summary = get_summary(text)
    return title, date, article, summary

def write_file(id, url, title, date, content, summary, target_path):
    json_dict = {}
    json_dict['id']=id
    json_dict['url']=url
    json_dict['title']=title
    json_dict['date']=date
    json_dict['content']='\n'.join(content)
    json_dict['summary']=summary

    with open(f"{target_path}/{id}.json", 'w') as json_file:
        json.dump(json_dict, json_file)

def proceed_one(text, url, path):
    id = get_id(url)
    title, date, article, summary = extract_data(text)
    write_file(id, url, title, date, article, summary, path)

def post_process(html, resp, **kwargs):
    """Function to process the gathered response from the request"""
    if resp.status == 200:
        proceed_one(html, str(resp.url), path=kwargs.get("target_path"))
        return ""
    else:
        return "Request failed"

counter = 0
print("Reading file in", args.url_path)
read_urls = pd.read_csv(args.url_path)
read_urls.drop_duplicates(inplace=True)
all_urls = read_urls["urls"].to_list()
prog = 0
# print("Finding current progress...")
# try:
#     with open("progress.txt", "r") as f:
#         prog = int(f.read())
# except:
#     prog = 0
# print("Current progress is", prog)
n = int(args.batch)
chunked_url = [all_urls[i:i + n] for i in range(prog, len(all_urls), n)]
# epoch = 0

print("Visit python main.py -h to get help on all arguments can be used")
print("="*50)
print(f"[WARNING] This program will scrap {len(all_urls)-prog} urls in {len(chunked_url)} iteration")
sure = input("Are you sure? [y/n]")
print("="*50)

if sure == "y":
    for read_url in chunked_url:
        async_Scrape = AsyncScrape(
            post_process_func=post_process,
            post_process_kwargs={"target_path": args.target_path},
            fetch_error_handler=None,
            use_proxy=False,
            proxy=None,
            pac_url=None,
            consecutive_error_limit=100,
            attempt_limit=5,
            rest_between_attempts=True,
            rest_wait=15,
            randomise_headers=True,
        )
        res = async_Scrape.scrape_all(read_url)
        # prog += len(read_url)
        # epoch += 1
        # if epoch >= int(args.epoch):
        #     print("Progress: ", prog)
        #     break
        time.sleep(int(args.rest_time))

    # with open("progress.txt", "w") as f:
    #     f.write(str(prog))