[poc_repository]: https://github.com/raghavsikaria/Project-Corynorhinus
[elon_tweets_dataset]: https://www.kaggle.com/datasets/andradaolteanu/all-elon-musks-tweets?select=TweetsElonMusk.csv
[sentence_transformers_model]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[sentence_transformers]: https://www.sbert.net/index.html
[analysis_jupyter_notebook]: https://www.sbert.net/index.html
[topic_jupyter_notebook]: https://www.sbert.net/index.html
[maarten_tds]: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
[umap_blog]: https://pair-code.github.io/understanding-umap/
[project_blog]: https://raghavsikaria.github.io/project-find-the-similar-tweet/

# THE BLOG IS UP FOR THIS PROJECT - [PLEASE READ ABOUT IT HERE, IT CONTAINS THE DETAILED BLOG AND ALL INTERACTIVE VISUALIZATIONS][project_blog]

# Introduction

I recently came across this use-case: **"Given a huge dataset of tweets, can you find N contextually most relevant tweets to a given user tweet?"** 

[**NOTE:** This is an active on-going project]

My first thought was to find embeddings of all tweets in dataset, then compute cosine distance of each of the tweets in dataset with the given user tweet and then simply find the top N tweets with the highest similarity. [The code-base for the POC can be found [here][poc_repository]]
Surprisingly it does work well.

Limitations of this approach followed in POC:

+ Cosine similarity calculation and comparison with all other tweets in the dataset does not scale well
+ Even if I use Numpy library to calculate this metric, or implement one in Cython, this is still essentially calculating distance of that tweet with every tweet in the dataset, sorting out the distances and is thus a bottleneck. For a dataset with >>> millions tweets, this will definitely not scale well, especially if there are many such given user tweets for which we have to find those contextually relevant tweets.

In order to address the above limitations for the use case, I thought of clustering all of the tweet embeddings from the dataset, but the fact that there can be multiple contexts associated with a tweet means that a tweet can/should be a part of multiple clusters. The whole black-box nature of these embeddings gives me doubt. I don't have much experience with clustering, so I'm a bit clueless. I am thinking of trying this out and see what happens.

This project is my attempt at clustering our tweet embeddings from the dataset and trying to solve the use-case.

# About Dataset

Given the lack of good quality free datasets available on the net and time (since I wanted a quick POC), I chose to go ahead with [Elon Musk's Tweets from 2010-2021][elon_tweets_dataset] that I found on Kaggle. My sincere thanks to user _andradaolteanu_ and community for this dataset. It contains roughly ~12k tweets possibly covering all the tweets in that timeframe from Elon.

# About embeddings

HuggingFace to the rescue like always! I've used pre-trained models from [Sentence-Transformers][sentence_transformers], particularly [this model][sentence_transformers_model] considering its speed, effectiveness and the fact that it is recommended for Semantic Search and Textual Similarity tasks. For each given input it will yield a vector of 384 dimensions. Read more about its specifics directly from the link given.