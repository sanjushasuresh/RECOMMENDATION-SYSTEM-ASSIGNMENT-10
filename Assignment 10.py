# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:00:18 2022

@author: LENOVO
"""

import numpy as np
import pandas as pd

data=pd.read_csv("book.csv",encoding='latin-1')
data.columns
data=data.iloc[:,1:]
data.columns
data.shape
data.isnull().sum()
data.head()
data.duplicated
data=data.drop_duplicates(keep=False)  # Duplicates are dropped
data.shape

# Sorting the user ID
df=data.sort_values('User.ID')

df['User.ID'].unique()
len(df['User.ID'].unique())  # 2182 users are there

df['Book.Title'].value_counts()  
df["Book.Rating"].value_counts() # Highest rating 8 is given 2283 times
df = data.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')

df=df.fillna(value=0,axis=0)
df.head()

# Calculating cosine based similarties
from sklearn.metrics import pairwise_distances
df=1-pairwise_distances(df.values,metric='cosine')
df
df=pd.DataFrame(df)
df

# Changing the index name and columns name for better understanding the similarities b/w the users
df.index=data['User.ID'].unique()
df.columns=data['User.ID'].unique()
df.iloc[0:10,0:10]

# Same user id has similarity as 1 so i have replace them with 0
np.fill_diagonal(df.values,0)
df

# Checking the highest similarities btw the users
df.idxmax(axis=1)[0:10]

# Checking which similarities the users have
data[(data['User.ID']==276737) | (data["User.ID"]==276726)]
data[(data['User.ID']==276768) | (data["User.ID"]==276726)]
data[(data['User.ID']==276774) | (data["User.ID"]==278543)]
data[(data['User.ID']==276788) | (data["User.ID"]==276726)]
data[(data['User.ID']==276729) | (data["User.ID"]==276726)]
data[(data['User.ID']==276772) | (data["User.ID"]==1491)]
data[(data['User.ID']==276748) | (data["User.ID"]==161677)]
data[(data['User.ID']==276744) | (data["User.ID"]==276726)]
data[(data['User.ID']==276796) | (data["User.ID"]==276726)]
# Here in every similarity each user gave similar ratings for a book and  
# there are no users who have read the same book