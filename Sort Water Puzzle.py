#!/usr/bin/env python
# coding: utf-8

# # API data

# ## Install libraries

# In[2]:


get_ipython().system('pip install google_play_scraper')


# In[3]:


get_ipython().system('pip install app_store_scraper')


# ## Import libraries

# In[7]:


from google_play_scraper import app, Sort, reviews_all
from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import json, os, uuid


# ## API users' review from CH Play and AppStore

# In[8]:


g_reviews = reviews_all(
        "sort.water.puzzle.pour.color.tubes.sorting.game",
        sleep_milliseconds=0, # mặc định là 0
        lang='en', # mặc định là 'vi'
        country='us', # mặc định là 'us'
        sort=Sort.NEWEST, # mặc định là Sort.MOST_REELEVANT
    )
a_reviews = AppStore('us', 'sort-water-color-puzzle', '1575680675')
a_reviews.review()


# In[9]:


g_df = pd.DataFrame(np.array(g_reviews),columns=['review'])
g_df2 = g_df.join(pd.DataFrame(g_df.pop('review').tolist()))
 
g_df2.drop(columns={'userImage', 'reviewCreatedVersion'},inplace = True)
g_df2.rename(columns= {'score': 'rating','userName': 'user_name', 'reviewId': 'review_id', 'content': 'review_description', 'at': 'review_date', 'replyContent' : 'developer_response', 'repliedAt': 'developer_response_date', 'thumbsUpCount': 'thumbs_up'},inplace = True)
g_df2.insert(loc=0, column='source', value='Google Play')
g_df2.insert(loc=3, column='review_title', value=None)
g_df2['laguage_code'] = 'en'
g_df2['country_code'] = 'us'
g_df2


# In[10]:


a_df = pd.DataFrame(np.array(a_reviews.reviews),columns=['review'])
a_df2 = a_df.join(pd.DataFrame(a_df.pop('review').tolist()))

a_df2.drop(columns={'isEdited'},inplace = True)
a_df2.insert(loc=0, column='source', value='App Store')
a_df2['developer_response_date'] = None
a_df2['thumbs_up'] = None
a_df2['laguage_code'] = 'en'
a_df2['country_code'] = 'us'
a_df2.insert(loc=1, column='review_id', value=[uuid.uuid4() for _ in range(len(a_df2.index))]) 
a_df2.rename(columns= {'review': 'review_description','userName': 'user_name', 'date': 'review_date','title': 'review_title', 'developerResponse': 'developer_response'},inplace = True)
a_df2 = a_df2.where(pd.notnull(a_df2), None)
a_df2


# ## Concatenating results between CH Play and AppStore

# In[32]:


result2 = pd.concat([g_df2,a_df2])
result2


# ## Save result to csv file

# In[33]:


df = pd.DataFrame(result2)  # Chuyển đổi dữ liệu đánh giá thành DataFrame
df.to_csv('result2.csv', index=False)  # Lưu DataFrame vào file CSV


# In[35]:


df_read = pd.read_csv('result2.csv')  # Đọc file CSV
print(df_read.head())  # Hiển thị một số hàng đầu tiên của DataFrame để kiểm tra


# In[36]:


result2


# # Sentiment Analysis with Python

# Sentiment Analysis: the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns

plt.style.use('ggplot')

import nltk


#  ## Step 1. Read in Data and NLTK Basics

# In[8]:


# Read in Data
df = pd.read_csv("result2.csv")
print(df.shape)


# In[9]:


df.info()


# In[10]:


df.head(50)


# ### Drop unused columns

# In[11]:


columns_to_drop = ['thumbs_up', 'developer_response', 'developer_response_date', 'appVersion', 'laguage_code', 'country_code']

df.drop(columns=columns_to_drop, inplace=True)


# In[12]:


df.head(100)


# ### Add ID columns for dataset

# In[13]:


df['ID'] = range(1, len(df) + 1)
df.head(100)


# In[14]:


id_column = df.pop('ID')
df.insert(0, 'ID', id_column)
df.head(100)


# ### Quick EDA

# In[37]:


df.info()


# In[38]:


sns.color_palette("RdBu_r")
ax = sns.countplot(x=df["rating"])
ax.set(xlabel='Rating', ylabel='Number of Reviews',
       title='Number of reviews per Rating group')

for p in ax.patches:
    txt = str(((p.get_height()/15721 )*100).round(2)) + '%'
    txt_x = p.get_x() + 0.1
    txt_y = p.get_height() +5
    ax.text(txt_x,txt_y,txt)

plt.show()


# ### Data transformation

# In[107]:


df['review_description_lower'] = df['review_description'].astype(str).str.lower()
df.head(5)


# ### Extracting relevant features for topic modelling

# In[41]:


tm_df = df[["review_title", "review_description", "review_description_lower", "rating"]]
tm_df.head(5)


# ## Step 2. Topic modeling using LDA

# ## converting reviews to document term matrix

# In[45]:


# discards words that occur in more than 97% of documents, and include words that occur at least in 2 documents
cv = CountVectorizer(max_df=0.97, min_df=2, stop_words='english')
doc_term_matrix = cv.fit_transform(tm_df['review_description_lower'])
doc_term_matrix


# ### hyperparameter tuning using gridsearch

# In[46]:


lda=LatentDirichletAllocation(random_state=101, n_jobs=-1)
param_grid = { 
    'n_components': [3, 4, 5, 10, 15],
    'max_iter': [5, 10, 15, 20, 25],
    'learning_decay': [.5, .7, .9]
}


# In[63]:


LDA=LatentDirichletAllocation(random_state=101, n_jobs=-1, learning_decay = 0.5, max_iter = 25, n_components = 3)
LDA.fit(doc_term_matrix)


# In[64]:


for index,topic in enumerate(LDA.components_):
    print(f'topic #{index} : ')
    print([cv.get_feature_names_out()[i] for i in topic.argsort()[-30:]])


# In[53]:


tm_df = tm_df.reset_index(drop=True)


# In[66]:


topic = LDA.transform(doc_term_matrix)
df_topic = pd.DataFrame(topic, columns=[
'0_GameDifficulty',
'1_GamePlay' ,
'2_GamingExperience'
])
df = pd.merge(tm_df, df_topic,  how='inner', left_index=True, right_index=True )


# In[74]:


def return_topic(row):
    if row["1_GamePlay"] > row["0_GameDifficulty"] and row["1_GamePlay"] > row["2_GamingExperience"]:
        return "GamePlay"
    if row["0_GameDifficulty"] > row["1_GamePlay"] and row["0_GameDifficulty"] > row["2_GamingExperience"]:
        return "GameDifficulty"
    if row["2_GamingExperience"] > row["0_GameDifficulty"] and row["2_GamingExperience"] > row["1_GamePlay"]:
        return "GamingExperience"


# In[75]:


df["review_topic"] = df.apply(lambda x: return_topic(x), axis=1)


# In[76]:


df.head(5)


# In[77]:


ax = sns.countplot(data=df, y="review_topic", hue="rating")
ax.set(xlabel='Number of Reviews', ylabel='Issue Topic',
       title='Distribution of Ratings per Topic')

plt.show()


# It is easy to see that the number of 5-star reviews is overwhelming compared to the rest of the reviews. People seems to be quite satisfied with all features.
# We can see that features for gaming experience created the highest number of reviewers, follow by game difficulty and lastly gameplay for facilities.

# ## Step 3. Wordcloud of positive and negative reviews

# In[24]:


# !pip install -q contractions


# In[81]:


import nltk
import re
from nltk.corpus import stopwords
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


# In[82]:


df.head(5)


# ### Data cleaning

# In[83]:


def clean_text(text):
    text = text.lower()
    # remove \n \t and non-alphanumeric
    text = re.sub("(\\t|\\n)", " ", text)
    text = re.sub("[^a-zA-Z']", " ", text)
    text = re.sub("(?:_|[^a-z0-9_:])[;:=]+[\)\(\-\[\]\/|DPO]", "", text)
    text = re.sub("[0-9]+", "", text)
    text = text.strip()
    # expanding the contractions
    text = ' '.join([contractions.fix(x) for x in text.split(" ")])
    return text.strip()

df["review_description_lower"] = df["review_description_lower"].apply
(lambda x: clean_text(x))


# In[95]:


#data pre-processing - LEMMATIZATION
def lemma_preprocess_text(text_list):
    processed_text = []
     
    #Tokenize words
    tokens = [word_tokenize(text) for text in text_list]
    
    #Remove stop words
    stop_list = stopwords.words('english')
    stop_list.append("filler")
    text_stop = [[word for word in doc if word.lower() not in stop_list] for doc in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text_lemma = [[lemmatizer.lemmatize(word) for word in doc] for doc in text_stop]
    
    return text_lemma


# In[109]:


# Put each row of the 'review_description' column into a list on its own
text_list = df['review_description_lower'].tolist()

# Preprocess the text_list
lemma_processed_text = lemma_preprocess_text(text_list)

# Add the processed_text list as a new column in the dataframe
df['lemmatized_processed_text'] = lemma_processed_text

# Preview the processed data
df.head(5)


# In[111]:


# reviews with rating >= 4
four_and_five = df[['rating'] >= 4]

# reviews with rating < 4
less_than_4 = df[df['rating'] < 4]


# ### Normal wordcloud

# In[112]:


# 4&5 RATINGS  - word cloud

# extract the processed text column as a list of lists

four_and_five_text = four_and_five['lemmatized_processed_text'].tolist()

# join each inner list into a single string
four_and_five_rev_text = [" ".join(x) for x in four_and_five_text]

# join the list of strings into a single string
four_and_five_rev_text = " ".join(four_and_five_rev_text)

# tokenize the words
four_and_five_rev_tokens = word_tokenize(four_and_five_rev_text)

# create a word cloud
review_wordcloud = WordCloud(width = 3000, height = 2000, background_color="white").generate(" ".join(four_and_five_rev_tokens))

# plot the word cloud
plt.figure(figsize=(8,6))
plt.imshow(review_wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Positive Reviews")
plt.axis("off")
plt.show()


# In[113]:


# LESS THAN 4 RATINGS  - word cloud

# extract the processed text column as a list of lists

less_than_4_text = less_than_4['lemmatized_processed_text'].tolist()

# join each inner list into a single string
less_than_4_rev_text = [" ".join(x) for x in less_than_4_text]

# join the list of strings into a single string
less_than_4_rev_text = " ".join(less_than_4_rev_text)

# tokenize the words
less_than_4_rev_tokens = word_tokenize(less_than_4_rev_text)

# create a word cloud
# wordcloud = WordCloud().generate(" ".join(tokens))
review_wordcloud = WordCloud(width = 3000, height = 2000, background_color="black").generate(" ".join(less_than_4_rev_tokens))

# plot the word cloud
plt.figure(figsize=(8,6))
plt.imshow(review_wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Negative Reviews")
plt.axis("off")
plt.show()


# Most of the words are indeed related to the game difficulty and users' experience: fun, good, game, level, ads, etc. Some negative words are more related to the users experience with the game: easy, ads, level, boring, best, etc.

# ## Step 4. VADER Sentiment Scoring
# 

# We will use NLTK's SentimentIntesityAnalyzer to get the neg/neu/pos scores of the text.
# This uses a "bag of words" approach:
#     1. Stop words are removed
#     2. Each word is scored and combined to a total score

# In[16]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# ### Run the polarity score on the entire dataset

# In[17]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    review = row['review_description']
    myid = row['ID']
    res[myid] = sia.polarity_scores(review)


# ### Add the result to the dataset

# In[18]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'ID'})
vaders = vaders.merge(df, how='left')


# In[19]:


vaders.head(10)


# In[20]:


vaders.describe()


# ### Plot VADER results

# In[23]:


vaders.boxplot(column=['compound'], by='rating').set(xlabel='Rating')
plt.title('') 
plt.ylabel('Compound Sentiment Scores')
plt.show()


# In[22]:


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='rating', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='rating', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='rating', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# It confirms what I hope to see and shows that vader is valuable in having this connection between the score of the text and sentiment rating. It relates to the actual rating review of the users. We can see that positivity is higher as the rating is higher in terms of stars and vice versa.
