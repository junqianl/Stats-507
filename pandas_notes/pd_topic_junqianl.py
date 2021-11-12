#!/usr/bin/env python
# coding: utf-8

# ## Topic in Pandas

# **Junqian Liu**  
# *junqianl@umich.edu*

# In[2]:


import numpy as np
import pandas as pd


# ## Idioms - if/then

# ## Dataframe method
# - The dataframes allows us to change the values of one or more columns directly by the conditions
# - df.loc allows us to choose which columns to work as the condition, and which columns to be changed based on the conditions
# - More specifically, it works as df.loc[conditions, target columns] = values

# In[3]:


df = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df.loc[df.apple >= 5, "boy"] = -1
df.loc[df.apple >= 5, ["cat", "dog"]] = 555
df


# ## Pandas method
# - pandas also can achieve the same aim by setting up a mask
# - pandas.DataFrame.where allows to decide if the conditions are satisfied and then change the values
# - overall, the goal is achieved by setting up the mask to the dataframe and using pandas.DataFrame.where to replace the values.
# - needs to assign to the dataframe after replacing values

# In[4]:


df2 = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df_mask = pd.DataFrame(
    {"apple": [True] * 4, "boy": [False] * 4, "cat": [True, False] * 2, 
    "dog": [False] * 4}
)
df2 = df2.where(df_mask, 818)
df2


# ## Numpy method
# - Similar to pandas method, np.where can also replace the value through if/then statement
# - It is more convenience as it doesn't need to set up the masks
# - It works by np.where(condistions, if true, else), to be more specific, the example is given below

# In[5]:


df3 = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df3["elephant"] = np.where(df["apple"] > 5, "water", "banana")
df3

