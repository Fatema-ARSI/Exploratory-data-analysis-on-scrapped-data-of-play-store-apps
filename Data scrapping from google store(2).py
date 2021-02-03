#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install google-play-scraper


# In[1]:


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


# In[2]:


# choose some apps that fit the crieteria from the healthcare and fitness demain
app_packages=['com.h8games.helixjump',
             'io.voodoo.crowdcity',
              'io.voodoo.paperio',
              'com.PupInteractive.TwistyRoad',
              'com.btstudios.twenty48sol',
              'com.nomonkeys.ballblast',
             'com.itchmedia.ta3',
             'io.voodoo.dune',
             'fr.two4tea.fightlist',
              'com.NikSanTech.FireDots3D']
              


# In[139]:


#scrape the infos for each app
app_infos = []

for ap in tqdm(app_packages):
    info=app(ap,lang='en',country='us')
    del info['comments']
    app_infos.append(info)


# In[4]:


app_infos


# In[5]:


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


# In[140]:


#save app information into pandas 
df_app = pd.DataFrame(app_infos)
df_app.head()


# # Data Manipulation

# In[53]:


#drop the unnecessary columns:
df_app.drop(['descriptionHTML','summaryHTML','sale','minInstalls', 'saleTime','originalPrice','saleText','offersIAP',
         'inAppProductPrice','developerId','developerEmail','developerWebsite','developerAddress','privacyPolicy',
             'developerInternalID','genreId','headerImage','screenshots','video','videoImage',
             'contentRatingDescription','adSupported','recentChangesHTML','url'],axis=1,inplace=True)


# In[54]:


##convert installs to numeric
df_app['installs']=df_app['installs'].replace('[\+\,]','',regex=True)


# In[55]:


df_app['installs'] = df_app['installs'].str.replace(u'\xa0', u'',regex=True).astype('int')


# In[48]:


df_app['released']=df_app['released'].apply(lambda x:str(x).replace(' ','-'))


# In[42]:


#check the datatype of variables
df_app.dtypes


# In[46]:


#see the distribution of score
sns.distplot(df_app['score'],kde=True,hist=True,color='darkblue',bins=10)


#  All the gaming apps have the rating aboove 3.6 with the highest count on 4.3 followed by  3.7

# In[141]:


#lets see the games by their year of release
year=pd.DatetimeIndex(df_app['released']).year # extract the year of released date
year=year.sort_values()
colors = ['rgb(204,229,255)','rgb(153,204,255)','rgb(102,178,255)','rgb(51,153,255)','rgb(0,128,255)','rgb(0,128,255)',
          'rgb(0,128,255)','rgb(0,102,204)','rgb(0,76,153)','rgb(0,51,102)']

data = {'year' : year, 'Color' : colors}
df_by_year=pd.DataFrame(data)
df_by_year['title']=df_app['title']


# In[142]:


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

# In[57]:


##lets see the installation per app
fig = px.histogram(df_app, x="title",y='installs',title='Installation by App')
fig.show()


# Games like Helix Jump and Crowd City have high installs compared to others making the data skewed,to understand the pattern of the data better apply log transformation

# In[58]:


df_app['loginstalls']=np.log(df_app['installs'])


# In[59]:


fig = px.histogram(df_app, x="title",y='loginstalls',title='Installation by App')
fig.show()


# In[60]:


## correalation between variables
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df_app.corr(),annot=True,cmap='winter') 


# Installs is postively correlated by 82% with ratings of game and by 83% with reviews,but also reviews and ratings are correlated by 100%. Hence we cannot use both the varible together to make the prediction on high installation. 

# In[62]:


data=df_app[['installs','reviews']]
fig=px.scatter(data,x='installs',y='reviews',title='The Relationship between reviews and Installation')
fig.show()


# In[84]:


plt.figure(figsize=(10,8))
rat_int = df_app.groupby(['ratings'])['installs'].mean().sort_values()
sns.barplot(x=rat_int, y=rat_int.index, data=df_app)


# It is clear that installation  are higher when the games have higher reviews and ratings . so inorder to increase the installation of games, one must concentrate on increasing its ratings.

# In[113]:


# scrapping the app reviews
app_reviews=[]

for ap in tqdm(app_packages):
    for score in range(1,6):
        for sort_order in [Sort.MOST_RELEVANT,Sort.NEWEST] :
            rvs=reviews(ap,lang='fr',country='fr',sort=sort_order,count=200 if score==3 else 100,
                        filter_score_with=score)[0]
            for r in rvs:
                r['sortOrder']='most_relevant' if sort_order==Sort.MOST_RELEVANT else 'newest'
                r['appId']=ap
            app_reviews.extend(rvs)


# In[116]:


#save app information into pandas 
df_reviews = pd.DataFrame(app_reviews)
df_reviews.head(1)


# In[117]:


#length of app reviews
len(app_reviews)


# In[118]:


#mMerge the df_app and df_reviews on appId
df=pd.merge(df_app,df_reviews,on='appId')


# In[119]:


df.head(1)


# In[120]:


#Drop the columns which are of no use
df.drop(['description','summary','installs','score_x','ratings',
         'reviews','histogram','size','androidVersion','androidVersionText','developer','genre','icon',
         'contentRating', 'containsAds','released','updated','version','recentChanges','appId'],
        axis=1,inplace=True)


# In[121]:


df.head(1)


# In[122]:


df['score_y'].value_counts()


# In[123]:



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


# In[124]:


#add the column sentiment which will contain values like positive,negative and neutral
def to_sentiment(rating):
    rating=int(rating)
    if rating<=2:
        return 'Negative'
    elif rating==3:
        return 'Neutral'
    else:
        return 'Positive'


# In[125]:


df['sentiment']=df.score_y.apply(to_sentiment)


# In[126]:


class_names=['Negative','Neutral','Positive']


# In[127]:



ax=sns.countplot(df.sentiment)
ax.set_xticklabels(class_names);


# It is clear most of the app have positive reviews

# In[128]:


# lets see the sentiments value by app
sentiments = df.groupby(['title','sentiment']).sentiment.count().unstack()
sentiments.plot(kind='bar',title='Sentiments by App')


# 2048 zen cards has the high positive sentiments compared less negative sentiments followed by  twisty_road whereas its clear other games has equal sentiment counts of positive,negative and neutral.

# Lets understand how many times customer reviews gets answered from the developer

# In[129]:



reviews_reply=df.groupby(['title']).agg({'content': ['count'],'replyContent': [ 'count'] })
reviews_reply['replynot']=((reviews_reply['content']-reviews_reply['replyContent'])/(reviews_reply['content']))*100
reviews_reply


# It is clear that developer does not reply to any of the reviews on games.

# In[130]:


#distribution of replyNot by app
repliedNot=reviews_reply['replynot']

repliedNot=reviews_reply['replynot']
repliedNot.plot(kind='bar',title='Percentage of replyNot by app')


# In[131]:


df.to_csv(r'C:\\Users\\fatem\\Desktop\\projects\\app_reviews.csv',index=False)


# In[ ]:




