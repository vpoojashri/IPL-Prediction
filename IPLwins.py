#!/usr/bin/env python
# coding: utf-8

# In[128]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[129]:


st.set_page_config(layout="wide")


# In[130]:


st.title("IPL WINNING PREDICTION")


# In[131]:


deliveries=pd.read_csv(r"C:\Users\ADMIN\Downloads\deliveries.csv")


# In[132]:


matches=pd.read_csv(r"C:\Users\ADMIN\Downloads\matches.csv")
matches.head()


# In[133]:


matches["winner"]=matches["winner"].replace("Rising Pune Supergiants","Rising Pune Supergiant")


# In[134]:


df = deliveries.merge(matches, how = 'inner', left_on='match_id', right_on='id')
print(df.shape)
df.head(2)


# In[135]:


df=df.drop_duplicates()


# In[136]:


df.city.fillna('INFO_MISSING', inplace = True)
df.umpire1.fillna('INFO_MISSING', inplace = True)
df.umpire2.fillna('INFO_MISSING', inplace = True)

df.player_of_match.fillna('INFO_MISSING', inplace = True)
df.player_dismissed.fillna('Not Applicable', inplace = True)
df.dismissal_kind.fillna('Not Applicable', inplace = True)
df.fielder.fillna('Not Applicable', inplace = True)


# In[137]:


df=df.drop(columns=["umpire3","id"])
df=df.dropna()


# In[138]:


fig=plt.figure(figsize=(5,4))
sns.barplot(y=df["winner"],x=df["total_runs"])
st.subheader("Total Runs Occured By Each Team")
st.pyplot(fig)


# In[139]:


plt.figure(figsize=(20,8))
d=matches.groupby(matches["season"])["winner"].value_counts()
year=[]
winner=[]
count=[]
for i in d.iteritems():
        year.append(i[0][0])
        winner.append(i[0][1])
        count.append(i[1])


# In[140]:


figure,axis=plt.subplots(figsize=(15,5))
figure.autofmt_xdate()

sns.barplot(x=winner,y=count,hue=year)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
st.subheader("Winning Count of Each Team Over Years")
st.pyplot(figure)


# In[141]:


venue=df["venue"].value_counts()[:5]
count=venue.tolist()
venue1=["M Chinnaswamy Stadium","Eden Gardens","Feroz Shah Kotla","Wankhede lStadium","MA Chidambaram Stadium, Chepauk"]


# In[142]:


fig=plt.figure(figsize=(6,4))
sns.barplot(y=venue1,x=count)
st.subheader("Top Stadium Where Most Matches Are Played")
st.pyplot(figure)


# In[143]:


fig=plt.figure(figsize=(5,3))
df_n=matches[matches['toss_winner']==matches['winner']]
slices=[len(df_n),(577-len(df_n))]
labels=['yes','no']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0.05),autopct='%1.1f%%' ,colors=['red','yellow'])

plt.show()
st.subheader("Predicting Toss Winner Becomes Winner")
st.pyplot(fig)


# In[144]:


plt.figure(figsize=(8,10))
sns.countplot(y=df['venue'],hue=df['toss_decision'],palette = 'YlGn')


# In[145]:


fig=plt.figure(figsize=(5,3))
d=df.groupby(df["batsman"])["batsman_runs"].sum()
d_sort=d.sort_values(ascending=False)[:5]
d_sort.plot(kind="barh",color="maroon")
st.subheader("Top 5 Batsman")
st.pyplot(fig)


# In[146]:


total_6s = df[df["batsman_runs"] == 6].groupby("season")["batsman_runs"].agg(['count'])

total_4s=df[df["batsman_runs"] == 4].groupby("season")["batsman_runs"].agg(['count'])


# In[147]:


fig=plt.figure(figsize=(5,3))
plt.plot(total_4s,color='blue',marker='o')
plt.plot(total_6s,color='red',marker='o')

plt.legend(["4_s","6_s"])
st.subheader("Total 4_s and 6_s")
st.pyplot(fig)


# In[148]:


total_6s_batsman = df[df["batsman_runs"] == 6].groupby("batsman")["batsman_runs"].count()
t6=total_6s_batsman.sort_values(ascending=False)[:5]
total_4s_batsman = df[df["batsman_runs"] == 4].groupby("batsman")["batsman_runs"].count()
t4=total_4s_batsman.sort_values(ascending=False)[:5]


# In[149]:


fig=plt.figure(figsize=(5,3))
t6.plot(kind="barh",color="red")
st.subheader("Top 5 batsman hit 4_s")
st.pyplot(fig)


# In[151]:


fig=plt.figure(figsize=(5,3))
t4.plot(kind="barh")
st.subheader("Top 5 batsman hit 6_s")
st.pyplot(fig)


# In[152]:


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
    


# In[153]:


fig=plt.figure(figsize=(8,3))
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


# In[154]:


df["winner"].unique()


# In[155]:


team=['Sunrisers Hyderabad', 'Rising Pune Supergiant',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians',
       'Delhi Daredevils', 'Gujarat Lions', 'Chennai Super Kings',
       'Rajasthan Royals', 'Deccan Chargers', 'Pune Warriors',
       'Kochi Tuskers Kerala']


# In[156]:


def prediction(team1,team2):
    p=(df[(df["team1"]==team1) & (df["team2"]==team2)])
    p1=(p["winner"].value_counts().sort_values(ascending=False))
    q=(df[(df["team1"]==team2) & (df["team2"]==team1)])
    q1=(q["winner"].value_counts().sort_values(ascending=False))
    if (len(p1)==2) or (len(q1)==2):
        if len(p1)==2:
            print(p1.index[0])
        elif len(q1)==2:
            print(q1.index[0])
    if (df[df["winner"]==team1].count()[0]) > (df[df["winner"]==team2].count()[0]):
        return(team1,"Has High chance of Winning")
    else:
        return(team2,"Has High chance of Winning") 

user_input1=st.selectbox("enter team1 ",team)
user_input2=st.selectbox("enter team2",team)
result=prediction(user_input1,user_input2)
st.subheader("PREDICTION")
st.title(result)


# In[ ]:




