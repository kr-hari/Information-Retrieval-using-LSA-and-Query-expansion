# Information Retrieval using LSA and Query expansion - Natural Language Processing (NLP)

This repository contains Information Retrieval system developed using [Latent Semantic Analysis](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9) and [Query expansion methods using WordNet thesaurus](https://www.mitpressjournals.org/doi/abs/10.1162/coli.2006.32.1.13). 

## Features
- Implementation of 4 Informatrion Retrieval algorithms
-- Latent Semantic Analysis (LSA)
-- Query expansion using WordNet thesaurus
-- Combination of LSA and query expansion
-- TFIDF (could be used as benchmark for other algorithms)

- Multiple ways to implement word processing including tokenization, sentence segmentation and stop word removal. The program has been made extremely modular keeping this point.
- Usage of external corpus for training. 20newsgroups corpus has been implemented.
- Automatically evaluates you results using Cranfield dataset queries (evaluation for other datasets can be easily implemented) using evaluation metrics such as Precision, Recall, MAP, F-score and nDCG metrics for top_n = 1 to 10. 

## Setting Up 
- Clone the repository to any folder using
    ```s
    $ git clone https://github.com/Harikr16/Information-Retrieval-using-LSA-and-Query-expansion/upload
    ```
- Install the required libraries using 
    ```s
    $ pip install -r requirements.txt
    ```

## Usage
- If you are using  for the first time , copy the cranfield dataset into a folder. Run :
    ```s
    $ python main.py --corpus=cranfield --restrict_tokenization=False --dataset=<path_to_cranfield_dataset>
    ```
    
##### Please feel free to improve the code. To contact me -  harikrishnankr16@gmail.com

