#!/usr/bin/env python
# coding: utf-8

# In[25]:


import streamlit as st


# In[26]:


st.title("IPL WINNING PREDICTION")


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


deliveries=pd.read_csv(r"C:\Users\ADMIN\Downloads\deliveries.csv")


# In[29]:


matches=pd.read_csv(r"C:\Users\ADMIN\Downloads\matches.csv")
matches.head()


# In[30]:


matches["winner"]=matches["winner"].replace("Rising Pune Supergiants","Rising Pune Supergiant")


# In[31]:


df = deliveries.merge(matches, how = 'inner', left_on='match_id', right_on='id')
print(df.shape)
df.head(2)


# In[32]:


df=df.drop_duplicates()


# In[33]:


df.city.fillna('INFO_MISSING', inplace = True)
df.umpire1.fillna('INFO_MISSING', inplace = True)
df.umpire2.fillna('INFO_MISSING', inplace = True)

df.player_of_match.fillna('INFO_MISSING', inplace = True)



df.player_dismissed.fillna('Not Applicable', inplace = True)
df.dismissal_kind.fillna('Not Applicable', inplace = True)
df.fielder.fillna('Not Applicable', inplace = True)


# In[34]:


df=df.drop(columns=["umpire3","id"])
df=df.dropna()


# In[35]:


fig=plt.figure(figsize=(3,3))
sns.barplot(y=df["winner"],x=df["total_runs"])
st.subheader("Total Runs Occured By Each Team")
st.pyplot(fig)


# In[36]:


plt.figure(figsize=(20,8))
d=matches.groupby(matches["season"])["winner"].value_counts()
year=[]
winner=[]
count=[]
for i in d.iteritems():
       year.append(i[0][0])
       winner.append(i[0][1])
       count.append(i[1])


# In[37]:


figure,axis=plt.subplots(figsize=(15,5))
figure.autofmt_xdate()

sns.barplot(x=winner,y=count,hue=year)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
st.subheader("Winning Count of Each Team Over Years")
st.pyplot(figure)


# In[38]:


fig=plt.figure(figsize=(6,8))
sns.countplot(y=df["venue"],order=df['venue'].value_counts(ascending=False).index)
st.pyplot(fig)


# In[39]:


fig=plt.figure(figsize=(3,3))
df_n=matches[matches['toss_winner']==matches['winner']]
slices=[len(df_n),(577-len(df_n))]
labels=['yes','no']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0.05),autopct='%1.1f%%' ,colors=['red','yellow'])

plt.show()
st.subheader("Predicting Toss Winner Becomes Winner")
st.pyplot(fig)


# In[40]:


plt.figure(figsize=(8,10))
sns.countplot(y=df['venue'],hue=df['toss_decision'],palette = 'YlGn')


# In[41]:


fig=plt.figure(figsize=(3,3))
d=df.groupby(df["batsman"])["batsman_runs"].sum()
d_sort=d.sort_values(ascending=False)[:5]
d_sort.plot(kind="barh",color="maroon")
st.subheader("Top 5 Batsman")
st.pyplot(fig)


# In[42]:


total_6s = df[df["batsman_runs"] == 6].groupby("season")["batsman_runs"].agg(['count'])

total_4s=df[df["batsman_runs"] == 4].groupby("season")["batsman_runs"].agg(['count'])


# In[43]:


fig=plt.figure(figsize=(3,3))
plt.plot(total_4s,color='blue',marker='o')
plt.plot(total_6s,color='red',marker='o')

plt.legend(["4_s","6_s"])
st.subheader("Total 4_s and 6_s")
st.pyplot(fig)


# In[44]:


total_6s_batsman = df[df["batsman_runs"] == 6].groupby("batsman")["batsman_runs"].count()
t6=total_6s_batsman.sort_values(ascending=False)[:5]
total_4s_batsman = df[df["batsman_runs"] == 4].groupby("batsman")["batsman_runs"].count()
t4=total_4s_batsman.sort_values(ascending=False)[:5]


# In[45]:


fig=plt.figure(figsize=(3,3))
t6.plot(kind="barh",color="red")
st.subheader("Top 5 batsman hit 4_s")
st.pyplot(fig)


# In[ ]:


fig=plt.figure(figsize=(3,3))
t4.plot(kind="barh")
st.subheader("Top 5 batsman hit 6_s")
st.pyplot(fig)


# In[46]:


runs=df.groupby(["season","batsman"])["batsman_runs"].sum()
raina=[]
kohli=[]
warner=[]
sharma=[]
gambir=[]
for i in runs.iteritems():
  if i[0][1]=="SK Raina":
    raina.append(i[1])
  if i[0][1]=="V Kohli":
    kohli.append(i[1])
  if i[0][1]=="DA Warner":
    warner.append(i[1])
  if i[0][1]=="RG Sharma":
    sharma.append(i[1])
  if i[0][1]=="G Gambhir":
    gambir.append(i[1])  
    


# In[47]:


fig=plt.figure(figsize=(5,5))
years=[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
sns.lineplot(y=raina,x=years)
sns.lineplot(y=kohli,x=years)
sns.lineplot(y=warner,x=[2009,2010,2011,2012,2013,2014,2015,2016,2017])
sns.lineplot(y=sharma,x=years)
sns.lineplot(y=gambir,x=years)
plt.plot(color=['red','blue','#772272','green','#FFAF00'],marker='o', figsize = (16,8))

plt.legend(["Raina","kohli","Warner","Sharma","Gambhir"])
st.subheader("top players")
st.pyplot(fig)


# In[65]:


df["winner"].unique()


# In[66]:


team=['Sunrisers Hyderabad', 'Rising Pune Supergiant',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians',
       'Delhi Daredevils', 'Gujarat Lions', 'Chennai Super Kings',
       'Rajasthan Royals', 'Deccan Chargers', 'Pune Warriors',
       'Kochi Tuskers Kerala']


# In[67]:


def prediction(team1,team2):
     if (df[df["winner"]==team1].count()[0]) > (df[df["winner"]==team2].count()[0]):
        return(team1,"Has High chance of Winning")
     else:
        return(team2,"Has High chance of Winning") 


user_input1=st.selectbox("enter team1 ",team)
user_input2=st.selectbox("enter team2",team)
result=prediction(user_input1,user_input2)
st.subheader("PREDICTION")
st.title(result)


# In[58]:





# In[ ]:


# option = st.selectbox('How would you like to be contacted?',('Email', 'Home phone', 'Mobile phone'))
# team1=st.write('You selected:', option)
# team2=st.write(option)

# st.write(prediction(team1,team2))

