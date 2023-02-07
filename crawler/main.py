import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd
from async_scrape import AsyncScrape
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser() 
parser.add_argument("-s", "--save_every", help = "save every n step(s)", default=1000)
parser.add_argument("-ts", "--target_start", help = "the initialization of liputan6 id", default=304259)
parser.add_argument("-te", "--target_end", help = "the initialization of liputan6 id", default=305000)
args = parser.parse_args()

def post_process(html, resp, **kwargs):
    """Function to process the gathered response from the request"""
    if resp.status == 200:
        soup = BeautifulSoup(html, 'lxml')
        url = soup.find_all(property="og:url")[0].get("content")
        resp_url = str(resp.url)
        id_news = [s for s in resp_url.split("/") if s.isdigit()]
        if id_news:
            if url:
                if str(id_news[0]) in url:
                    return url
        return np.nan
    else:
        return "Request failed"

target = [int(args.target_start)+i*int(args.save_every) for i in range(int((int(args.target_end) - int(args.target_start))/int(args.save_every))+1)]

for t in range(len(target)-1):
    async_Scrape = AsyncScrape(
        post_process_func=post_process,
        post_process_kwargs={},
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
    res = async_Scrape.scrape_all([
                f"https://www.liputan6.com/news/read/{id}"
                for id in range(target[t], target[t+1])
            ])

    df = pd.DataFrame(res)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["func_resp"], inplace=True)
    df.rename(columns={"func_resp": "urls"}, inplace=True)
    df["urls"].to_csv(f"results/{str(int(datetime.now().timestamp()))}.csv", index=False)
    time.sleep(5)