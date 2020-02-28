 -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:38:35 2020

@author: user
"""
#Tokenizing the Data
from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

##print(tweet_tokens)


#Normalizing the Data
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

#def lemmatize_sentence(tokens):
#    lemmatizer = WordNetLemmatizer()
#    lemmatized_sentence = []
#    for word, tag in pos_tag(tokens):
#        if tag.startswith('NN'):
#            pos = 'n'
#        elif tag.startswith('VB'):
#            pos = 'v'
#        else:
#            pos = 'a'
#        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
#    return lemmatized_sentence

#print(lemmatize_sentence(tweet_tokens[0]))


# Removing Noise from the Data
import re, string

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#print(remove_noise(tweet_tokens[0], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
#print(positive_tweet_tokens[500])
#print(positive_cleaned_tokens_list[500])
    
    
    
    
# Determining Word Density

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)  

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10))  




#Converting Tokens to a Dictionary

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)




#Splitting the Dataset for Training and Testing the Model

import random

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]






#Building and Testing the model
from nltk.tokenize import word_tokenize

from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

#print("Accuracy is:", classify.accuracy(classifier, test_data))
#
#print(classifier.show_most_informative_features(10))

import json
with open('review.json') as f:
  reviews = json.load(f)

review_lst =[]
for i in range(len(reviews)):
    review_lst.append(reviews[i]["comments"])

    
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

review_tokens = []
for r in review_lst:
    review_tokens.append(word_tokenize(r))

review_cleaned_tokens_list = []

for tokens in review_tokens:
    review_cleaned_tokens_list.append(remove_noise(tokens, stop_words))



review_neg_pos=[]
from nltk import classify

for i in range(len(review_cleaned_tokens_list)):
    custom_tokens = review_cleaned_tokens_list[i]
    review_neg_pos.append(classifier.classify(dict([token, True] for token in custom_tokens)))




no_of_pos_comments=0

for i in range(len(review_cleaned_tokens_list)):
    if review_neg_pos[i]=='Positive':
        no_of_pos_comments += 1
    
no_of_neg_comments=len(review_cleaned_tokens_list) - no_of_pos_comments
#print(no_of_pos_comments,no_of_neg_comments)
        
#My Data Set



import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Positive', 'Negative'
sizes = [no_of_pos_comments, no_of_neg_comments]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



# Word Plot


from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt





class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
    
    
color_to_words = {
    # words below will be colored with a green single color function
    '#00ff00': [],
    # will be colored with a red single color function
    'red': []
}    
default_color = 'grey'

All_Text=""
from nltk.stem import PorterStemmer

ps = PorterStemmer()

for j in range(len(review_cleaned_tokens_list)):
    custom_token = review_cleaned_tokens_list[j]
    for k in range(len(custom_token)):
        custom_tokens = ps.stem(custom_token[k])
        All_Text = All_Text + ' '+ custom_tokens
        if classifier.classify(dict([token, True] for token in custom_tokens)) == 'Positive':
            color_to_words['#00ff00'].append(custom_tokens)
        else:
            color_to_words['red'].append(custom_tokens)
color_to_words['red'] = list(dict.fromkeys(color_to_words['red']))
color_to_words['#00ff00'] = list(dict.fromkeys(color_to_words['#00ff00']))       
wc = WordCloud(collocations=False).generate(All_Text.lower())

grouped_color_func = GroupedColorFunc(color_to_words, default_color)

# Apply our color function
wc.recolor(color_func=grouped_color_func)

# Plot
plt.figure()
plt.figure(figsize=[10,10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig("iphone7_flipkart.png", format="png")
plt.show()
