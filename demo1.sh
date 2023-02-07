#!/bin/sh

# install all requirements
pip install -r requirements.txt

# download liputan6_data
cd summarization
gdown --id 1LNRkeSVkTpS7-o8Go7vk6YZNVZ1eHZFh

# download summarization indolem-bertshare 150000 checkpoint
gdown --id 1Q9_LLhPhbboOPQ-u8aw8dzFYIMtijXKC

# download indobart fine-tuned paraphrase generation checkpoint
cd ..
cd paraphrase
gdown --id 1XyRIW4O0T4GeglufIpRobpP3hX1nbFNd