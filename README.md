# DISASTER TWEETS CLASSIFICATION

This project involves classifying the tweets as Disaster or Non-disaster tweets.

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. Hence, there arises the need to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

Moreover, this project/challenge has enabled me to understand about Natural Language Processing, Recurrent Neural Networks and their types as well as about Sequence Modelling problems briefly.

## About the Dataset:
**Natural Language Processing with Disaster Tweets** is an open competition in Kaggle which runs indefinitely with a rolling leaderboard.

The dataset used is publicly available in the Kaggle, and can be accessed here : [Dataset](https://www.kaggle.com/competitions/nlp-getting-started/data)

## Steps involved:
### 1. Data Loading and Pre-processing
In this step, I had first imported the necessary libraries and aftert hat loaded the dataset - train.csv and test.csv

Also, thorough visualisation of text and targets was done to get familiar with the data to be used for modelling purposes.

After that, I had further divided/splited the whole train data into train data (train_sentences and train_labels) and validation data (valid_sentences, valid_labels)

### 2. Text Vectorization and Embedding

**Text Vectorization** - A preprocessing layer which maps text features to integer sequences.

**Embedding** - A process of turning positive integers (indexes) into dense vectors of fixed size which can be updated while training the models. (Similar to weights).

In this step, I had set up the text vectorizer and embedding layer with proper required parameters like max_tokens, output_sequence_length, input_dim, ect and fit (adapt) both of them to the training sentences.

### 3. Modelling Experiments

**Model 0** (Baseline model) - Naive-Bayes with TF-IDF.

**Model 1** - Simple dense model

**Model 2** - Long-Short Term Memory (LSTM)

**Model 3** - Gated Recurrent Unit (GRU)

**Model 4** - Bidirectional Recurrent Neural Network

**Model 5** - Convolutional 1D

**Model 6** - Universal Sentence Encoder (Pre-trained embedding)

**Model 7** - Universal Sentence Encoder trained on 10% of data

### Insights:

While I was performing the modelling experiments, I got to learn about the way to visually understand the embeddings. 

Tensorflow has a dedicated tool called as [**Embedding Projector**](https://projector.tensorflow.org/) which allows us to visually see the embedding weights the model has learned or updated. This tool required two files namely - **vectors.tsv** and **metadata.tsv**(optional).

**The following image shows the Embedding weights of Simple Dense Model**

![image](https://github.com/Parekh03/Disaster_tweets_classification/assets/110220047/e715a1e0-24f7-4558-8c2a-e1763cdf8a32)

Moreover, I got to understand about the concept of **Data Leakage** while I was creating a 10% subset of training data for Universal Sentence Encoder (Model 7), and the need to resolve it.

## 4. Results
The Universal Sentence Encoder performs the best among the other models when evaluated on the following evaluation metrics:

**Accuracy - 82%**

**Precision - 82%**

**Recall - 76%**

**F1 score - 79%**

![image](https://github.com/Parekh03/Disaster_tweets_classification/assets/110220047/8dc5e1f0-e752-48ef-b569-b6b2647ae9f8)

## 5. Saving the best model

Saving the Universal Sentence Encoder in both the formats (HDF% and SavedModel format) for future use.

## 6. Making Predictions using the saved model

**Making predictions on random samples from the test data**

![image](https://github.com/Parekh03/Disaster_tweets_classification/assets/110220047/157fefa6-aa36-428f-9ade-12cbc539ffcf)


**Making predictions on real tweets**


![image](https://github.com/Parekh03/Disaster_tweets_classification/assets/110220047/c61f67e4-50f2-4b2d-a17a-46545185b242)
![image](https://github.com/Parekh03/Disaster_tweets_classification/assets/110220047/9a41a437-34d5-40eb-b689-e715b819a4f3)

**The Universal Sentence Encoder model works pretty well on the real data (tweets from the wild)**

## References/Resources

**I am really thankful to the following mentioned resources which have enabled me to learn about Introductory Natural Language Processing and have helped me build various NLP Models with conceptual clarity**
1. [Udemy - Tensorflow Developer Certificate](https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/?kw=tensorflow+developer+certif&src=sac)
2. [MIT's video lecture on NLP](https://youtu.be/ySEx_Bqxvvo?si=-2BwahPRE4imhwiP)
3. [Tensorflow's documentation](https://www.tensorflow.org/)
4. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
5. [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)





