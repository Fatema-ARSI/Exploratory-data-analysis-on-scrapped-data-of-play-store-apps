#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from google_play_scraper import Sort, reviews, app
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set(style='whitegrid', palette='muted', font_scale=1.2)


# In[2]:


# choose some apps that fit the crieteria from the healthcare and fitness demain
app_packages=['co.rythm.dreem',
              'com.dreem.dreemthoughts',
              'com.northcube.sleepcycle',
              'com.fitbit.FitbitMobile',
              'com.alltrails.alltrails',
              'fr.cnamts.it.activity',
              'nic.goi.aarogyasetu',
              'org.iggymedia.periodtracker',
              'com.runtastic.android',
              'homeworkout.homeworkouts.noequipment']


# In[3]:


#scrape the infos for each app
app_infos = []

for ap in tqdm(app_packages):

    info = app(ap, lang='en', country='us')

    del info['comments']

    app_infos.append(info)


# In[4]:


#got all 10 apps.lets write a helper fuction that prints JSON objects a bit better
def print_json(json_object):

    json_str = json.dumps(

    json_object,

    indent=2,

    sort_keys=True,

    default=str)
    print(highlight(json_str, JsonLexer(), TerminalFormatter()))


# In[5]:


print_json(app_infos[0])


# In[7]:


# create a plot of their icons
def format_title(title):

    sep_index = title.find(':') if title.find(':') != -1 else title.find('-')

    if sep_index != -1:

        title = title[:sep_index]

    return title[:10]

fig, axs = plt.subplots(2, len(app_infos) // 2, figsize=(14, 5))


for i, ax in enumerate(axs.flat):

    ai = app_infos[i]

    img = plt.imread(ai['icon'])

    ax.imshow(img)

    ax.set_title(format_title(ai['title']))

    ax.axis('off')


# In[8]:


#save app information into pandas and then into CSV file
app_infos_df = pd.DataFrame(app_infos)
app_infos_df.to_csv('apps.csv', index=None, header=True)


# In[10]:


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


# In[11]:


print_json(app_reviews[0])


# In[12]:


#length of app reviews
len(app_reviews)


# In[13]:


#save the file to pandas
df_app_reviews=pd.DataFrame(app_reviews)
df_app_reviews.head()


# In[20]:


#save dataframe to CSV
df_app_reviews.to_csv(r'D:\\Fatema\\Data Analyst\\Dataset\\app_reviews.csv',index=None,header=True)

