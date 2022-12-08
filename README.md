# CourseProject

This ReadMe is the documentation for the CS410 class project.

## Group Info

| Number | Name | ID |
| :---: | :---: | :---: |
| 1 | Michael Bencsik | bencsik2 |

## Free Topic

The goal was to classify news headlines and/or articles as either having positive or negative sentiment using the increase or decrease of the market price as the model's label.  

## Overview

### Data Clean Up

The `Data Clean` notebook demonstrates the dataset clean up that was required before processing the datasets. This notebook is split into 3 parts.

The first part will clean text by removing a lot of misc. characters and punctuation that was found in most of the datasets, remove stop words, remove extra spaces, and lower the case of the characters. Then, a new file will be saved with `_clean` added to the end of the file name.

![CleanData](./Pictures/CleanData.png)

To use, set the flag to `True`, change the `cleanFilePath` to your csv dataset location, and set `columnsToClean` to a list of the text (string) columns to clean up. Then, run the notebook.

The second part creates an individual stock dataset by combining a very large financial news dataset and a stock price dataset. The large financial news dataset was too large to upload to GitHub.

![SingleStockDataset](./Pictures/SingleStockDataset.png)

I will attempt to link the news dataset from my Google Drive. The dataset can be downloaded from [Kaggle - US Financial News](https://www.kaggle.com/datasets/gennadiyr/us-equities-news-data?resource=download). The individual stock prices were downloaded from [Nasdaq Historical](https://www.nasdaq.com/market-activity/quotes/historical). The code filters the news by stock ticker, creates a binary label for the stock price based on if the day is positive or negative, then combines the data into a single dataset to be used for the model. Set the flag to `True`, select the stock ticker from the list shown, then choose whether to use the news article title or content. Run the notebook to create a new csv dataset for the individual stock.

The last part of the notebook combines 25 columns of news titles from a Reddit stock market dataset from [Kaggle - Reddit Dow Jones](https://www.kaggle.com/datasets/aaron7sun/stocknews). This was done to create one data point per label. Set the flag to `True` and run the notebook to create the combined dataset.

![RedditCombine](./Pictures/RedditCombine.png)

 - - -

### Models

The two final models that were detail enough to present are based on TensorFlow's text classification tutorials [RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn) and [BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert). The original source code has been modified greatly to use my datasets allow for easy customization of the models features. These models are found in the `Sentiment - TF BERT` and `Sentiment - TF RNN` notebooks.

The initial dataset setup is similar in both notebooks, since they use the same datasets.

![ChooseDataset](./Pictures/ChooseDataset.png)

Select a stock ticker symbol or the Reddit dataset from the list shown. Select whether to use the news article title or content. The notebooks use the datasets that were created from the `Data Clean` notebook.

Next the `SHUFFLE_SEED` can be changed to shuffle the dataset. The seed is there in case the user would like to reproduce certain results.

![Seed](./Pictures/Seed.png)

The user can choose how to split the datasets into training, validation, and testing sets. The batch size can also be adjusted.

![TrainTest](./Pictures/TrainTestSplit.png)

### BERT

For the BERT model, a preprocessed BERT model needs to be selected. Expand the cell and select a model provided by TensorFlow.

![BERTPreprocess](./Pictures/BERTPreprocessed.png)

L2 regularization can be adjusted to smooth the losses of the model in training. This can help with over fitting.

![Regularization](./Pictures/Regularization.png)

Select which optimizer to use by setting `OPTIMIZER` to the name of the optimizer listed in the dictionary. Any optimizer from [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) can be used. It just needs to be added to the dictionary. Select the number of epochs to run by changing `epochs`. If the optimizer uses a learning rate, then set `init_lr` to the desired rate.

![Optimizer](./Pictures/Optimizer.png)

Then run the notebook! I would suggest using a small number for `epochs` as the BERT model can take a few minutes to run.

### RNN

Show differences of RNN notebook
Save model for demo?
Show how to run on Colab? Add flag for Colab or Local machine?





2. What is your free topic?
3. Your documented source code and main results.
    1. An overview of the function of the code (i.e., what it does and what it can be used for). 2) Documentation of how the software is implemented with sufficient detail so that others can have a basic understanding of your code for future extension or any further improvement. 3) Documentation of the usage of the software including either documentation of usages of APIs or detailed instructions on how to install and run a software, whichever is applicable. 
4. Self-evaluation.
    1. Have you completed what you have planned?
    2. Have you got the expected outcome?

    After cleaning the datasets, I started with various tutorials to use PyTorch, TensorFlow, Sklearn, and NLTK. 


    3. If not, discuss why.
5. A demo that shows your code can actually run and generate the desired results.
    - If there is a training process involved, you don’t need to show that process during the demo. If your code takes too long to run, try to optimize it, or write some intermediate results (e.g. inverted index, trained model parameters, etc.) to disk beforehand.
