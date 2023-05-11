import re
from pathlib import Path
import nltk
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
from gensim.models import Word2Vec
from joblib._multiprocessing_helpers import mp
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

# nltk.download('wordnet')
# nltk.download('stopwords')

# simple statistical analysis of dataset
'''
df = pd.read_excel('news_dataset.xlsx')
pandas.set_option('display.max_columns', None)
print('describe:\n', df.describe(include='all'))
df.info()
'''

# read dataset and merge categories
'''
# read news dataset, drop link, date, and author columns as it isn't useful data for modelling
df = pd.read_excel('news_dataset.xlsx',usecols=['category', 'headline', 'short_description'],
                   dtype={'category': str, 'headline': str, 'short_description': str})

# merge categories
df['category'] = df['category'].replace(['ARTS', 'ARTS & CULTURE', 'CULTURE & ARTS'], 'ART')
df['category'] = df['category'].replace(['WORLDPOST', 'THE WORLDPOST', 'WORLD NEWS'], 'WORLD NEWS')
df['category'] = df['category'].replace(['PARENTS', 'PARENTING'], 'PARENTS')
df['category'] = df['category'].replace(['ENVIRONMENT', 'GREEN'], 'ENVIRONMENT')
df['category'] = df['category'].replace(['WELLNESS', 'HEALTHY LIVING'], 'HEALTH')
'''


def upsample_df(df):
    # set max entries
    max_entries = 5000

    # set categories as list
    categories = df['category'].unique()

    # initialise empty upsampled dataframe
    upsampled_df = pd.DataFrame()

    # iterate through list of categories
    for category in categories:

        # create new dataframe
        category_df = df[df['category'] == category]

        # get number of entries in dataframe
        n_category = len(category_df)

        # if number of entries is more than max_entries
        if n_category >= max_entries:

            # randomly select entries from dataframe to add to upsampled dataframe to limit
            upsampled_df = pd.concat([upsampled_df, category_df.sample(max_entries)])

        # if number of entries is less than max_entries
        else:

            # number of entries to duplicate
            n_repeats = int(np.ceil(max_entries / n_category))

            # duplicate entries
            upsampled_category_df = pd.concat([category_df] * n_repeats)

            # add to upsampled dataframe
            upsampled_df = pd.concat([upsampled_df, upsampled_category_df.sample(max_entries)])

    # return upsampled dataframe
    return upsampled_df


# normalisation function
def normalise(text):
    # make text lowercase
    text = text.lower()

    # remove anything that isn't a letter or space
    text = re.sub(r'[^a-zA-Z ]+', '', text)

    # remove extra spaces, tabs, and new lines
    text = " ".join(text.split())

    # return normalised text
    return text


# stemming function
def stem(text):

    # create tokens from text
    tokens = word_tokenize(text)

    # stem each token
    ps = PorterStemmer()
    required_words = [ps.stem(x) for x in tokens]

    # reconstruct stemmed sentence from stemmed tokens
    stemmed_sentence = ' '.join(required_words)

    # return stemmed sentence
    return stemmed_sentence


# lemmatization function
def lemmatize(text):

    # set variable used for lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()

    # creates tokens
    tokens = word_tokenize(text)

    # lemmatize each token
    required_words = [wordnet_lemmatizer.lemmatize(x, 'v') for x in tokens]

    # reconstruct lemmatized sentence from lemmatized tokens
    lemmatized_sentence = ' '.join(required_words)

    # returns lemmatized sentence
    return lemmatized_sentence


# stopword removal function
def remove_stopwords(text):

    # create tokens
    tokens = word_tokenize(text)

    # check if token is a stopword, removes token if it is a stopword
    required_words = [x for x in tokens if x not in stopwords.words('english')]

    # reconstruct sentence from tokens that weren't stopwords
    sentence_without_stopwords = ' '.join(required_words)

    # return sentence without stopwords
    return sentence_without_stopwords


# vectorization technique 1 - bag of words
def count_vectoriser(X_train, X_test):
    # create a CountVectorizer object
    cv = CountVectorizer(ngram_range=(1, 2))

    # fit and transform the training data
    X_train = cv.fit_transform(X_train)

    # transform the testing data
    X_test = cv.transform(X_test)

    # return X_train and X_test
    return X_train, X_test


# vectorisation technique 2 - tf idf
def tf_idf_vectorizer(X_train, X_test):
    # create a TfidfVectorizer object
    tfidf = TfidfVectorizer(ngram_range=(1, 1))

    # fit and transform the training data
    X_train = tfidf.fit_transform(X_train)

    # transform the testing data
    X_test = tfidf.transform(X_test)

    # return X_train and X_test
    return X_train, X_test


# vectorisation technique 3 - word2vec
def word2vec(X_train, X_test, vector_size=100):
    # train the Word2Vec model on the training data
    sentences = X_train.apply(word_tokenize)
    model = Word2Vec(sentences, min_count=1, vector_size=vector_size, window=10, alpha=0.01, epochs=50, sg=1)

    # generate embeddings for each sentence in X_train
    X_train_w2v = []
    for sentence in X_train:
        tokens = word_tokenize(sentence)
        embedding = [model.wv[token] for token in tokens if token in model.wv.key_to_index]
        if len(embedding) > 0:
            X_train_w2v.append(sum(embedding) / len(embedding))
        else:
            X_train_w2v.append(np.zeros(vector_size))

    # generate embeddings for each sentence in X_test
    X_test_w2v = []
    for sentence in X_test:
        tokens = word_tokenize(sentence)
        embedding = [model.wv[token] for token in tokens if token in model.wv.key_to_index]
        if len(embedding) > 0:
            X_test_w2v.append(sum(embedding) / len(embedding))
        else:
            X_test_w2v.append(np.zeros(vector_size))

    # return X_train and X_test as numpy arrays
    return np.array(X_train_w2v), np.array(X_test_w2v)


# ml model 1 - multinomial naive bayes
def multinomial_naive_bayes(X_train, X_test, y_train, y_test):
    # train multinomial naive bayes
    nb_model = MultinomialNB(alpha=0.05)
    nb_model.fit(X_train, y_train)

    # test model on train and test set
    y_train_pred = nb_model.predict(X_train)
    y_test_pred = nb_model.predict(X_test)

    # get training set accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # get testing set accuracy, f1 score, recall, precision
    test_accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    precision = precision_score(y_test, y_test_pred, average='weighted')

    # print evaluation metrics
    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("Precision: {:.2f}%".format(precision * 100))

    # get confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # labels for plotting confusion matrix as heatmap
    labels = np.unique(y_test)

    # calculate percentage of correct predictions
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    # convert confusion matrix into pandas dataframe
    cm_df = pd.DataFrame(cm_pct, index=labels, columns=labels)

    # plot confusion matrix as heatmap
    sns.heatmap(cm_df, annot=False, fmt='.2%', cmap='Blues', square=True, xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# ml model 2 - linear svm
def linear_svm(X_train, X_test, y_train, y_test):
    # train linear svm model
    svm_model = LinearSVC(C=0.05, tol=0.25, intercept_scaling=1)
    svm_model.fit(X_train, y_train)

    # test model on train and test set
    y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)

    # get training set accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # get testing set accuracy, f1 score, recall, precision
    test_accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    precision = precision_score(y_test, y_test_pred, average='weighted')

    # print evaluation metrics
    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("Precision: {:.2f}%".format(precision * 100))

    # get confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # labels for plotting confusion matrix as heatmap
    labels = np.unique(y_test)

    # calculate percentage of correct predictions
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    # convert confusion matrix into pandas dataframe
    cm_df = pd.DataFrame(cm_pct, index=labels, columns=labels)

    # plot confusion matrix as heatmap
    sns.heatmap(cm_df, annot=False, fmt='.2%', cmap='Blues', square=True, xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# ml model 3 - decision tree
def decision_tree(X_train, X_test, y_train, y_test):
    # train decision tree model
    dt = DecisionTreeClassifier(random_state=42, max_features=100000, min_samples_leaf=50)
    dt.fit(X_train, y_train)

    # test model on train and test set
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    # get training set accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # get testing set accuracy, f1 score, recall, precision
    test_accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    precision = precision_score(y_test, y_test_pred, average='weighted')

    # print evaluation metrics
    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("Precision: {:.2f}%".format(precision * 100))

    # get confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # labels for plotting confusion matrix as heatmap
    labels = np.unique(y_test)

    # calculate percentage of correct predictions
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    # convert confusion matrix into pandas dataframe
    cm_df = pd.DataFrame(cm_pct, index=labels, columns=labels)

    # plot confusion matrix as heatmap
    sns.heatmap(cm_df, annot=False, fmt='.2%', cmap='Blues', square=True, xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# normalisation, stemming, and lemmatization
'''
# drop rows with empty cells
df = df.dropna()

# normalise dataset
df = df.applymap(normalise)
print('normalisation complete')

# stem dataset
df = df.applymap(stem)
print('stemming complete')

# lemmatize dataset
df = df.applymap(lemmatize)
print('lemmatization complete')
'''

# multiprocess stopword removal, save dataframe as csv
'''
# drop null values
df = df.dropna()

# split the headline and short_description columns into 16 chunks of equal size
n_chunks = 16
chunk_size = len(df) // n_chunks
chunks_headline = [df['headline'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
chunks_short_desc = [df['short_description'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]


# function to apply remove_stopwords to each chunk in parallel
def process_chunk(chunk):
    return chunk.apply(remove_stopwords)


if __name__ == '__main__':
    mp.freeze_support()
    # apply 'process_chunk' function to each chunk of the 'headline' column using multiprocessing
    with mp.Pool(processes=n_chunks) as pool:
        results_headline = pool.map(process_chunk, chunks_headline)

    # combine the results of the headline column into a single dataframe
    df['headline'] = pd.concat(results_headline)

    # apply process_chunk function to each chunk of the short_description column using multiprocessing
    with mp.Pool(processes=n_chunks) as pool:
        results_short_desc = pool.map(process_chunk, chunks_short_desc)

    # combine the results of the 'short_description' column into a single dataframe
    df['short_description'] = pd.concat(results_short_desc)


# drop rows with empty cells
# df = df.dropna()

# drop duplicates if there are any
# df.drop_duplicates(inplace=True)

# save processed data to csv
filepath = Path('C:/Users/Ilya/PycharmProjects/Machine_Learning_Assignment/news_dataset_two.csv')
df.to_csv(filepath, index=False)
'''

# news_data_set_two is the preprocessed dataset
df = pandas.read_csv('news_dataset_two.csv')

# drop null values
df = df.dropna()

# category merging that was missed before preprocessing
df['category'] = df['category'].replace(['style beauti', 'style'], 'style')
df['category'] = df['category'].replace(['food drink', 'tast'], 'taste')

# combine headline and short description
news_data = df['headline'] + ' ' + df['short_description'] # combine 'headline' and 'short_description'
category = df['category']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(news_data, category, train_size=0.8, random_state=42)

# rebalance classes
df = upsample_df(df)
print('class imbalance correction complete')

# do vectorisation on x train and x test (tf idf, count, word2vec)
X_train, X_test = count_vectoriser(X_train, X_test)
# X_train, X_test = tf_idf_vectorizer(X_train, X_test)
# X_train, X_test = word2vec(X_train, X_test)
print('vectorisation complete')

# use vectors on ml model (bayes, linear, decision)
multinomial_naive_bayes(X_train, X_test, y_train, y_test)
# linear_svm(X_train, X_test, y_train, y_test)
# decision_tree(X_train, X_test, y_train, y_test)
