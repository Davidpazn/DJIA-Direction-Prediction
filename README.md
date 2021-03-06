# DJIA-Direction-Prediction

Kaggle Case: January 2022

## David Pacheco Aznar

## Daily News for Stock Market Prediction
The dataset was extracted from:
Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved [11/24/2021] from [Daily News for DJIA Prediction](https://www.kaggle.com/aaron7sun/stocknews)

## Requirements
In order to execute these notebooks, the txt file **requirements.txt** file has been provided.

It can be done as follows with conda:
**conda create --name djia**
and then,
**pip3 install -r requirements.txt**

That should be enough to have the environment ready to execute the sripts.

**INFO:** 
If Colab is being used, uncomment requirements.txt line: `talib-binary==0.4.19`, and comment `ta-lib==0.4.19`
Otherwise, it should be enough as it is.

## Demo 
Can be found in the Demo jupyter notebook.
Example results are:

![Screenshot from 2022-01-09 20-17-38](https://user-images.githubusercontent.com/70665433/148698828-fcf9f86a-2c55-47e1-9009-f2fd274266c7.png)

## Dataset Explanation and Objective
Find all the Exploratory Data Analysis: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZSGhf5dmb8uKFYgSmhCaaNStj_qXQnuP?usp=sharing)

In short, the whole objective is to predict the whether the stock market will go up or go down as compared to the previous day. For that, the dataset author provides the top 25 Reddit headlines every day from 2008-06-08 to 2016-07-01 and a label of whether the market went up or down. In addition, he also provides a time series with the top 25 news (even on non-trading days) and a time series extracted from Yahoo-Finance from 2008-08-08 to 2016-07-01.
The labels are encoded as: 1 if the market goes up or stays the same, 0 if it goes down. The headlines are not sorted by relevance, they are just the top 25 news in no particular order.
The market time series has an OHLCV format, that is: Open, High, Low, Close, Volume format.

![djia](https://user-images.githubusercontent.com/70665433/148307523-5ca9c275-4ca1-4226-ac1f-fe06642d9ffa.png)

## Preprocessing
#### Texts
Since we have text indexed to a time series, the first step has gone towards cleaning the texts. The texts have been cleaned using different methods. 
  1. Find ngrams (Gensim), detect nouns, verbs, adverbs, adjectives and proper nouns (SpaCy)
  2. Simple cleaning (Basic preprocessing for BERT embedder)
  3. Data Cleaning detecting nouns, verbs, adverbs, adjectives and proper nouns, but ignoring ngrams

These preprocessings have been used to find relations between words/n-grams and stock market moves, accomplishing relatively good results such as:

Ngrams that only appear when the market goes up:

![up_ngrams](https://user-images.githubusercontent.com/70665433/148307557-b147c520-d95b-49b8-8306-7ad7461cd59b.png)

Ngrams that only appear when the market goes down:

![down_ngrams](https://user-images.githubusercontent.com/70665433/148307568-a89bd96f-e116-4c18-a5e4-52a1a323f6ea.png)

Then, these preprocessings (Simple for BERT and ngrams), have been used to find topics from the news. After extensive exploration, no trivial features have been found to be relevant. Hence, the idea to cluster all news using different embeddings so as to find unsupervised topics in news. The optimal number of topics has been found by using a HDP (Hierarchical Dirichlet Process).
The result being a total of 150 topics and the best silhouette score (**0.39**) being achieved with BERT and a dimensionality reduction with Umap.

The clustering accomplished with Bert + Umap + preprocessing with SpaCy
![bert_umap](https://user-images.githubusercontent.com/70665433/148307631-5d203242-c9c7-4508-be68-cda76e24a237.png)

#### Time Series
The time series presents a clear upper trend until 2015. The seasonality comes to 0. However, by using **fractional differentiation** (de Prado), only a 0.6 factor of differentiation of the original adjusted close price series is needed in order to pass the dickey fuller test with a p-value lower than 0.005.
Fractional Differentiation vs p-value
![fracdiff](https://user-images.githubusercontent.com/70665433/148307582-533df726-4643-476c-a055-3fbc15725c9d.png)

Also, 25 indicators have been added to increase predicitve power.

## Modelling
### LSTM Modelling
Link to Model Code:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18s0oAwiHupkfJhUzqIVYdWUrsZSZm2Dk?usp=sharing)

The model is a stacked LSTM with up to four inputs built ontop of the tensorflow API.
The inputs are:
  1. Window of past n-returns (the fwd_return is dropped, since it would cause data leackage).
  2. Topics classified from BERT with Umap as dummy variables
  3. Sentiments on news (readability metrics, polarity...)
  4. Stock market indicators
With one layer and no tuning, the model accomplishes an AUC above 0.5 in the whole test-set. In addition, accomplishes a **54%** in out-of-bag samples.

![lstm_stack](https://user-images.githubusercontent.com/70665433/148307779-0d98d5ff-259b-400a-bf66-c73604dd5614.png)

Surprisingly, the best combination of inputs has been found to be Window of n-returns and Topics. That is, topics from BERT, made a better prediciton on stock price than stock indicators.

### Classic Models approach
Link to Classic Models code:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19RwtJIp1XLp1teisv-yfBdT9Vlx5QeqG?usp=sharing)

Since LSTM models got a good performance, the decision to test with non deep-learning models, was made. The results weren't that bad in the cv, in fact, some actually got 56 AUC. The feature importance was: ![image](https://user-images.githubusercontent.com/70665433/148698100-42c6c854-c9b7-433e-a0dd-2c6b5ad0766b.png)

Meaning, the most important features were the past returns, but to some extent, some topics, one sentiment and one indicator.  When performing K-Bests, the result were all 230 features for best performance. LGBM and RF have been runed using OPTUNA and skopt respectively. Finally, a Voting Classifier with hard and soft decision making were made. Resulting in around 55 AUC on the tet-set. Pretty good performance and faster than LSTMs.

## Tuning
Colab Link to Tuning :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lGf5cSIHqEAOur8kYxPhLLa0ERis9ESP?usp=sharing)

After the great success accomplished with a stacked LSTM, efforts have been made to optimize the neural network hyperparameters such as: the number of units per layer, layer dropouts, optimizer epsilon and learning rate and batch size. To avoid overffiting, keras-tuner base class Tuner has been overridden in order to add a Blocking Time Series Cross-Validation while optimizing parameters using Bayesian Optimization.
The results of the cross-validation vary, but top performing configurations get to up to 55-57% average AUC on test-sets. 
The following image represents how Blocking Time Series Cross-Validation works. The main idea is to avoid data leackage and is said to be better for cross-validation than standard cumulative cross-validation (Sklearn TimeSeriesSplit implemenatation).

![blocking_time_series_split](https://user-images.githubusercontent.com/70665433/148366364-7f71773b-f613-498d-864c-7004d86c32e5.png)

Train-sets in blue, test-sets in red.

Also, the sliding window option was also added, in order to be able to perform two different types of cv on time series. Both scipts (sliding window and blocking time series) can be found in _src/ts_utils/rnn.py_.

Classic and LSTM CV scores and time to run (google Colab)
 Model | Time (5 Folds CV) | AUC (Tuned) - CV
 :---: | :---: | :---: 
2 Layer LSTM| 1???15s| Up to 58.68 
3 Layer LSTM | 3???20s | Up to 57.28
Soft Voting Ensemble: LGBM + RF + XGB | 20s | Up to 56.60
Hard Voting Ensemble: LGBM + RF + XGB | 21s | Up to 58.25


## Conclusions
Access Model Comparison and Metric Analysis Notebook in colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_InNf404L16d-H-e7M4mUOEQvL2-WlzC?usp=sharing)

Finally, the metric analysis and model comparison. This last notebook, recaps all models built accross the notebook. Since no model seemed to be consistently outperforming a given threshold (or there were lots of bias towards a class), the decision to build a stacking classifier was done. That incorporates the top 2 performing models in the test set, in order to reduce variance and hopefully, improve the AUC. The results have been impressing, leaving a **62 AUC** and **65% accuracy** in out of bag data.
The ROC curves:

![image](https://user-images.githubusercontent.com/70665433/148698485-a8179835-2bdc-4b34-b876-c05a38fcd776.png)

The Precision Recall Curves:

![image](https://user-images.githubusercontent.com/70665433/148698495-053efac3-5159-4de5-858c-fc0ea0c737cb.png)

Finally, the confusion matrix of the stacking model:

![image](https://user-images.githubusercontent.com/70665433/148698511-b634f4c7-8688-4c64-95b8-55d9727d7ec5.png)


The model outperformed the "random" 50% chance up-down model and the always-buy method (out-of-bag enabled for a 56% accuracy with this strategy). 
When it ocmes to things to improve, data really lacked. More observations would have been very helpful to the model, since with standard train-test-split, there is exactly a breakpoint where the test-set begins and that may be causing problems in the prediction. However, the performance, in the end seems to be really good. 

## Future work
In order to improve performance, it could be interesting to try using more specific news. That is, industry specific news. In addition, single stock prediction or indices of industries could very probably be better suited for NLP models. Correlation would be higher and hence, better performance would very possibly be achieved. Also, as stated earlier, more data to train the model should have been presented. That is, in form of other stocks, higher frequency, a larger time frame or build a GAN to generate fake data to train with.
To improve the model, encoders and decoders could be a good option, also TCNs or other forms of convolutional/recurrent networks. Also, the stacking model could be further improved by using a neural net to join predictions instead of using XGBClassifier.

## License
This repository is licensed under GPL-2.0 License. See LICENSE for details.
