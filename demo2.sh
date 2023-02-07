#!/bin/sh

# download liputan6_data
cd summarization
tar -xvf liputan6_data.tar.gz

# download summarization indolem-bertshare 150000 checkpoint
tar -xvf checkpoint-150000.tar.gz

# download indobart fine-tuned paraphrase generation checkpoint
cd ..
cd paraphrase
cd save
tar -xvf filtered_paracotta.tar.gz