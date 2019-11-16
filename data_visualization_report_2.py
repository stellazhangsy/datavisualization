#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('/Users/stella/Downloads')


# In[14]:


import pandas as pd
df=pd.read_csv('master.csv')
df.head()


# In[11]:


df.info()


# In[40]:


df=df.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age',
                      'suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop',
                      'country-year':'CountryYear','HDI for year':'HDIForYear',
                      ' gdp_for_year ($) ':'GdpPerYear','gdp_per_capita ($)':'GdpPerCapital',
                      'generation':'Generation'})


# In[41]:


df.columns


# In[19]:


suicidesNO=[]
for country in df.Country.unique():
    suicidesNO.append(sum(df[df['Country']==country].SuicidesNo))


# In[20]:


suicidesNO = pd.DataFrame(suicidesNO,columns=['suicidesNO'])
country=pd.DataFrame(df.Country.unique(),columns=['country'])
dt_suicide=pd.concat([suicidesNO,country],axis=1)


# In[21]:


dt_suicide=dt_suicide.sort_values(by='suicidesNO',ascending=False)


# In[22]:


sns.barplot(y=dt_suicide.country[:30], x=dt_suicide.suicidesNO[:30])
plt.show
plt.xlabel('Total Number of Suicides from 1985-2016')
plt.ylabel('Country')


# In[23]:


my_pal = {"Generation X": "b", "Silent": "g", "G.I. Generation": "r", "Boomers": "c", "Millenials": "m", "Generation Z": "k"}


# In[24]:


sns.countplot(df.Generation, palette = my_pal)
plt.title('Number of Suicides by Generation')
plt.xticks(rotation=45)
plt.ylabel('Number of Suicides')
plt.show()


# In[27]:


list1=df[df.SuicidesNo>1000]
sns.swarmplot(x="Gender", y="SuicidesNo",hue="Generation", data=list1, palette = my_pal)
plt.title('Number of Suicides by Gender and Generation')
plt.ylabel('Number of Suicides')
plt.show()


# In[28]:


df['Age'].unique()


# In[29]:


index_population=[]
for age in df['Age'].unique():
    index_population.append(sum(df[df['Age']==age].Population)/len(df[df['Age']==age].Population))


# In[30]:


plt.bar(['15-24 years','35-54 years','75+ years','25-34 years','55-74 years','5-14 years'],index_population, align='center')
plt.xticks(rotation=45)
plt.show()


# In[37]:


fig=sns.jointplot(y='GdpPerCapital',x='SuicidesNo',kind='hex',data=df[df]])
plt.show()


# In[38]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[42]:


#suicides and GDP
byCountry = df.groupby('Country').mean().sort_values('Suicides100kPop', ascending=False).reset_index()
g=sns.jointplot(x='GdpPerCapital', y='Suicides100kPop',data=byCountry,kind='regression')


# In[43]:


#United States Case 
us=df[df['Country']=='United States']


# In[44]:


us_year = us.groupby('Year').mean().reset_index()
us_gender=us.groupby(['Gender','Year']).mean().reset_index
us_age=us.groupby(['Age','Year']).mean().reset_index


# In[45]:


g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=us_year, kind='regression')


# In[46]:


jp=df[df['Country']=='Japan']
jp_year = jp.groupby('Year').mean().reset_index()
jp_gender=jp.groupby(['Gender','Year']).mean().reset_index
jp_age=jp.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=jp_year, kind='regression')


# In[47]:


Ger=df[df['Country']=='Germany']
Ger_year = Ger.groupby('Year').mean().reset_index()
Ger_gender=Ger.groupby(['Gender','Year']).mean().reset_index
Ger_age=Ger.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=Ger_year, kind='regression')


# In[48]:


jm=df[df['Country']=='Jamaica']
jm_year = jm.groupby('Year').mean().reset_index()
jm_gender=jm.groupby(['Gender','Year']).mean().reset_index
jm_age=jm.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=jm_year, kind='regression')


# In[49]:


chile=df[df['Country']=='Chile']
chile_year = chile.groupby('Year').mean().reset_index()
chile_gender=chile.groupby(['Gender','Year']).mean().reset_index
chile_age=chile.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=chile_year, kind='regression')


# In[50]:


it=df[df['Country']=='Italy']
it_year = it.groupby('Year').mean().reset_index()
it_gender=it.groupby(['Gender','Year']).mean().reset_index
it_age=it.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=it_year, kind='regression')


# In[51]:


Serbia=df[df['Country']=='Serbia']
Serbia_year = Serbia.groupby('Year').mean().reset_index()
Serbia_gender=Serbia.groupby(['Gender','Year']).mean().reset_index
Serbia_age=Serbia.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=Serbia_year, kind='regression')


# In[52]:


Ecua=df[df['Country']=='Ecuador']
Ecua_year = Ecua.groupby('Year').mean().reset_index()
Ecua_gender=Ecua.groupby(['Gender','Year']).mean().reset_index
Ecua_age=Ecua.groupby(['Age','Year']).mean().reset_index
g = sns.jointplot(x="GdpPerCapital", y="Suicides100kPop", data=Ecua_year, kind='regression')


# In[53]:


import os


# In[54]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS


# In[55]:


df_word = df[df.SuicidesNo>1500]
plt.subplots(figsize = (9,9))
wordcloud= WordCloud(
    background_color = 'white',
                      width=480,
                      height=350
    ).generate(" ".join(df_word.Country))
plt.imshow(wordcloud)
plt.axis('off')
plt.margins(x=0,y=0)
plt.show()


# In[71]:


df2 = df[df['year'] != 2016]


# In[89]:


#gender
gender_df=pd.DataFrame({
    "Year":df2.year,
    "Gender":df2.sex,
    "SuicidesNo":df2.suicides_no
    
})


# In[90]:


gender_suicide=gender_df.pivot_table(index='Year',columns='Gender',aggfunc='mean')


# In[91]:


gender_suicide.head()


# In[92]:


x=gender_suicide.index
x


# In[93]:


gender_suicide.columns=['female','male']
gender_suicide.head()


# In[94]:


gender_suicide.tail()


# In[95]:


female=gender_suicide['female']
male=gender_suicide['male']


# In[96]:


fig,ax=plt.subplots(figsize=(10,5))
ax.plot(x, female, color="blue", alpha=0.5 , label='Female suicide rate')
ax.plot(x, male, color="green", alpha=0.5 , label="Male suicide rate")
ax.set_xlabel('Years')
ax.set_ylabel('Avg. of suicide no')
ax.legend()


# In[ ]:





# In[ ]:




