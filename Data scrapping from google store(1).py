#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install google-play-scraper
#pip  install Pygments 


# In[2]:


#import libraries

from google_play_scraper import Sort, reviews, app ##for scrapping app contents from google store
import pandas as pd
import numpy as np
from tqdm import tqdm

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.figure_factory as ff

#plotly.offline doesn't push your charts to the clouds
import plotly.offline as pyo
pyo.offline.init_notebook_mode()


# In[4]:


# choose some apps that fit the crieteria from the healthcare and fitness demain
app_packages=['co.rythm.dreem',
             'com.getsomeheadspace.android',
              'com.northcube.sleepcycle',
              'com.felix.calm',
              'com.processingbox.jevaisbiendormirpremium',
              'com.azumio.android.sleeptime.paid',
              'com.urbandroid.sleep.full.key',
             'ru.olegfilimonov.sleeptime',
             'com.processingbox.jevaisbiendormirpremium',
             'com.icechen1.sleepytime.plus']
              


# In[5]:


#scrape the infos for each app
app_infos = []

for ap in tqdm(app_packages):
    info=app(ap,lang='en',country='fr')
    del info['comments']
    app_infos.append(info)


# In[7]:


# create a plot of their icons
def format_title(title):
    sep_index = title.find(':') if title.find(':') != -1 else title.find('-')
    if sep_index != -1:
        title = title[:sep_index]
    return title[:20]

fig, axs = plt.subplots(2, len(app_infos) // 2, figsize=(14, 5))
for i, ax in enumerate(axs.flat):
    ai = app_infos[i]
    img = plt.imread(ai['icon'])
    ax.imshow(img)
    ax.set_title(format_title(ai['title']))
    ax.axis('off')


# In[195]:


#save app information into pandas 
df_app = pd.DataFrame(app_infos)
df_app.head(1)


# # Data Manipulation

# In[198]:


#drop the unnecessary columns:
df_app.drop(['descriptionHTML','summaryHTML','sale','minInstalls', 'saleTime','originalPrice','saleText','offersIAP',
         'inAppProductPrice','developerId','developerEmail','developerWebsite','developerAddress','privacyPolicy',
             'developerInternalID','genreId','headerImage','screenshots','video','videoImage',
             'contentRatingDescription','adSupported','recentChangesHTML','url'],axis=1,inplace=True)


# In[199]:


##convert installs to numeric
df_app['installs']=df_app['installs'].replace('[\+\,]','',regex=True).astype('int')


# In[200]:


#check the datatype of variables
df_app.dtypes


# In[210]:


#see the distribution of score
sns.distplot(df_app['score'],kde=True,hist=True,color='darkblue',bins=10)


#  All the apps have the rating aboove 3.5 with the highest count on 3.9 and 4.6

# In[203]:


#which app 
year=pd.DatetimeIndex(df_app['released']).year # extract the year of released date
year=year.sort_values()
colors = ['rgb(204,229,255)','rgb(153,204,255)','rgb(102,178,255)','rgb(51,153,255)','rgb(0,128,255)','rgb(0,128,255)',
          'rgb(0,128,255)','rgb(0,102,204)','rgb(0,76,153)','rgb(0,51,102)']

data = {'year' : year, 'Color' : colors}
df_by_year=pd.DataFrame(data)
df_by_year['title']=df_app['title']


# In[204]:


fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Title</b>' ,'<b>Year</b>'],
    line_color='white', fill_color='white',
    align='center', font=dict(color='black', size=12)
  ),
  cells=dict(
    values=[df_by_year.title, df_by_year.year],
    line_color=[df_by_year.Color], fill_color=[df_by_year.Color],
    align='center', font=dict(color='black', size=11)
  ))
])

fig.show()


# It will be interesting to find ot which factors aids in increasing the installation of the app. 

# In[205]:


##lets see the installation per app
fig = px.histogram(df_app, x="title",y='installs',title='Installation by App')
fig.show()


#  Apps like Headspace and Sleep Cycle have high installs compared to others making the data skewed,to understanf the dpattern of the data better apply log transformation

# In[206]:


df_app['loginstalls']=np.log(df_app['installs'])


# In[207]:


fig = px.histogram(df_app, x="title",y='loginstalls',title='Installation by App')
fig.show()


# In[211]:


## correalation between variables
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df_app.corr(),annot=True,cmap='winter') 


# Installs is postively correlated by 61% with type of app(free or not)  and by 65% with updated . May be this factors can make btter understanding on increasing the installation.

# In[212]:


data=df_app[['installs','updated']]
fig=px.scatter(data,x='installs',y='updated',title='The Relationship between Updated and Installation')
fig.show()


# In[215]:


fig=px.box(df_app,x='free',y='installs',color='free',title='Boxplot of type of apps')
fig.show()


# It is clear that installation  are higher when the apps free in which highest valus contains 10M installs and 50% of the apps have 500k installs. 

# In[216]:


# scrapping the app reviews
app_reviews=[]

for ap in tqdm(app_packages):
    for score in range(1,6):
        for sort_order in [Sort.MOST_RELEVANT,Sort.NEWEST] :
            rvs=reviews(ap,lang='en',country='us',sort=sort_order,count=200 if score==3 else 100,
                        filter_score_with=score)[0]
            for r in rvs:
                r['sortOrder']='most_relevant' if sort_order==Sort.MOST_RELEVANT else 'newest'
                r['appId']=ap
            app_reviews.extend(rvs)


# In[108]:


#save app information into pandas 
df_reviews = pd.DataFrame(app_reviews)
df_reviews.head(1)


# In[109]:


#length of app reviews
len(app_reviews)


# In[110]:


#mMerge the df_app and df_reviews on appId
df=pd.merge(df_app,df_reviews,on='appId')


# In[111]:


#for col in df.columns:
    #print(col)


# In[112]:


#Drop the columns which are of no use
df.drop(['description','summary','summaryHTML','installs','minInstalls','score_x','ratings',
         'reviews','histogram','size','androidVersion','androidVersionText','developer','genre','icon',
         'contentRating', 'containsAds','released','updated','version','recentChanges','appId'],
        axis=1,inplace=True)


# In[178]:


df.head(1)


# In[134]:



labels =  df['score_y'].value_counts().index
sizes = df['score_y'].value_counts().values
colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

explode = (0.1,0, 0, 0,0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, colors=colors_list,
autopct='%1.1f%%', shadow=True, startangle=140,pctdistance=1.12)

plt.axis('equal')
plt.title=('Ratio of Ratings')
plt.legend(labels=labels, loc='upper left') 
plt.show()


# In[118]:


#add the column sentiment which will contain values like positive,negative and neutral
def to_sentiment(rating):
    rating=int(rating)
    if rating<=2:
        return 'Negative'
    elif rating==3:
        return 'Neutral'
    else:
        return 'Positive'


# In[119]:


df['sentiment']=df.score_y.apply(to_sentiment)


# In[120]:


class_names=['Negative','Neutral','Positive']


# In[130]:



ax=sns.countplot(df.sentiment)
ax.set_xticklabels(class_names);


# It is clear most of the app have positive reviews

# In[340]:


# lets see the sentiments value by app
sentiments = df.groupby(['title','sentiment']).sentiment.count().unstack()
sentiments.plot(kind='bar',title='Sentiments by App')


# SleepyTime has the highest positive reviews followed by sleep_time and Relax&Keep_calm whereas its clear apps like Headspace, Sleep_cycle and sleep_as_android has equal sentiment counts of positive,negative and neutral.

# Lets understand how many times customer reviews gets answered from the developer

# In[331]:



reviews_reply=df.groupby(['title']).agg({'content': ['count'],'replyContent': [ 'count'] })
reviews_reply['replynot']=((reviews_reply['content']-reviews_reply['replyContent'])/(reviews_reply['content']))*100
reviews_reply


# SleepyTime plus  does not reply to their reviewers at all as well as Sleep Time does not reply to their reviewers 91.54% times. Dreem app replies to almost all the reviewers making it 28.57% times to repliedNot.

# In[339]:


#distribution of replyNot by app
repliedNot=reviews_reply['replynot']

repliedNot=reviews_reply['replynot']
repliedNot.plot(kind='bar',title='Percentage of replyNot by app')


# In[ ]:




