# Restaurant Reviews Sentiment Analysis

## Overview

This project focuses on sentiment analysis for restaurant reviews, allowing users to determine whether the feedback is positive or negative. The dataset consists of textual reviews, each labeled with sentiment (positive or negative). The primary goal is to create a tool that can automatically analyze and classify customer reviews based on their sentiment.

## Dataset

The dataset, stored in the "Restaurant_Reviews.tsv" file, includes two columns: "Review" and "Liked." The "Review" column contains the textual reviews, and the "Liked" column contains binary labels (1 for positive sentiment, 0 for negative sentiment).

## Preprocessing

Textual data undergoes preprocessing using natural language processing (NLP) techniques, such as:

- Removal of non-alphabetic characters.
- Conversion to lowercase.
- Tokenization into words.
- Removal of stopwords.
- Application of stemming using the Porter Stemmer.
- The processed data is then transformed into a Bag of Words representation using the CountVectorizer from scikit-learn.

## Model Training

The sentiment analysis model is trained using the Naive Bayes classifier, specifically the Gaussian Naive Bayes model. The dataset is split into training and testing sets for evaluating the model's effectiveness.


## Functionality

The core functionality of the program is to classify reviews into positive or negative sentiment. Users can leverage this tool to:

- Check Positive Feedback:

      Input a new review.
      Receive output indicating whether the sentiment is positive.

- Check Negative Feedback:

      Input a new review.
      Receive output indicating whether the sentiment is negative.

- Install the required dependencies:

```python
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
