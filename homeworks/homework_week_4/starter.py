#!/usr/bin/env python
# coding: utf-8

# In[1]:


#system('pip freeze | grep scikit-learn')


# In[2]:


#system('python -V')


# In[3]:


import pickle
import pandas as pd
import sys


# In[18]:

print('open the model bin')
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[ ]:


year = int(sys.argv[1])
month = int(sys.argv[2])


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df


# In[15]:
#year = 2022
#month = 2
print("reading the files")
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[16]:
print("doing the ml thing")

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)



# In[11]:


def result_df(df):
    df_2 = pd.DataFrame()
    df_2['ride_id'] = df['ride_id']
    df_2['predicted'] = y_pred

    return df_2


# In[13]:

print("creating the second table")
result_df = result_df(df)


# In[14]:


result = {'mean duration' : result_df['predicted'].mean(), 'std duration': result_df['predicted'].std()}
print(result)

# In[ ]:


#if __name__ == "__main__":
#    print(result)

