#!/usr/bin/env python
# coding: utf-8

# ## Topics in Pandas<br>
# **Stats 507, Fall 2021** 

# ## Contents<br>
# + [pandas.cut function](#pandas.cut-function) 
# + [Sampling in Dataframes](#Sampling-in-Dataframes)
# + [Idioms-if/then](#Idioms-if/then)

# ___<br>
# ## pandas.cut function<br>
# **Name: Krishna Rao**<br>
# <br>
# UM-mail: krishrao@umich.edu

# ## pandas.cut function<br>
# * Use the cut function to segment and sort data values into bins. <br>
# * Useful for going from a continuous variable to a categorical variable. <br>
# * Supports binning into an equal number of bins, or a pre-specified array of bins.<br>
# <br>
# #### NaNs?<br>
# * Any NA values will be NA in the result. <br>
# * Out of bounds values will be NA in the resulting Series or Categorical object.

# ## Examples<br>
# * Notice how the binning start from 0.994 (to accommodate the minimum value) as an open set and closes sharply at 10<br>
# * The default parameter 'right=True' can be changed to not include the rightmost element in the set<br>
# * 'right=False' changes the bins to open on right and closed on left

# In[2]:


import pandas as pd
import numpy as np
input_array = np.array([1, 4, 9, 6, 10, 8])
pd.cut(input_array, bins=3)
#pd.cut(input_array, bins=3, right=False)


# +<br>
# Observe how 0 is converted to a NaN as it lies on the open set of the bins<br>
# 1.5 is also converted to NaN as it lies between the sets (0, 1) and (2, 3)

# In[3]:


bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
#bins = [0, 1, 2, 3, 4, 5]
pd.cut([0, 0.5, 1.5, 2.5, 4.5], bins)
# -


# ## Operations on dataframes<br>
# * pd.cut is a very useful function of creating categorical variables from continous variables<br>
# * 'bins' can be passed as an IntervalIndex for bins results in those categories exactly, or as a list with continous binning.<br>
# * Values not covered by the IntervalIndex or list are set to NaN.<br>
# * 'labels' can be specified to convert the bins to categorical type variables. Default is `None`, returns the bins.

# ## Example 2 - Use in DataFrames<br>
# * While using IntervalIndex on dataframes, 'labels' can be updated with pd.cat.rename_categories() function<br>
# * 'labels' can be assigned as string, numerics or any other caregorical supported types

# +

# In[4]:


df = pd.DataFrame({"series_a": [0, 2, 1, 3, 6, 4, 2, 8, 10],
                   "series_b": [-1, 0.5, 2, 3, 6, 8, 14, 19, 22]})


# In[5]:


bin_a = pd.IntervalIndex.from_tuples([(0, 2), (4, 6), (6, 9)])
label_a = ['0 to 2', '4 to 6', '6 to 9']
df['bins_a'] = pd.cut(df['series_a'], bin_a)
df['label_a'] = df['bins_a'].cat.rename_categories(label_a)


# In[6]:


bin_b = [0, 1, 2, 4, 8, 12, 15, 19]
label_b = [0, 1, 2, 4, 8, 12, 15]
df['bins_b'] = pd.cut(df['series_b'], bin_b)
df['labels_b'] = pd.cut(df['series_b'], bin_b, labels=label_b)


# In[7]:


df
# -


# #### References:<br>
# * https://pandas.pydata.org/docs/reference/api/pandas.cut.html<br>
# * https://stackoverflow.com/questions/55204418/how-to-rename-categories-after-using-pandas-cut-with-intervalindex<br>
# ___

# ___<br>
# ## Sampling in Dataframes<br>
# **Name: Brendan Matthys** <br>
# <br>
# UM-mail: bmatthys@umich.edu

# ## Intro -- df.sample<br>
#     

# Given that this class is for an applied statistics major, this is a really applicable topic to be writing about. This takes a dataframe and returns a random sample from that dataframe. Let's start here by just importing a dataframe that we can use for 

# In[8]:


import pandas as pd
import os
import pickle
import numpy as np


# # +<br>
# -----------------------------------------------------------------------------

# In[9]:


filepath =os.path.abspath('')
if not os.path.exists(filepath + "/maygames"):
    nba_url ='https://www.basketball-reference.com/leagues/NBA_2021_games-may.html'
    maygames = pd.read_html(nba_url)[0]
    maygames = maygames.drop(['Unnamed: 6','Unnamed: 7','Notes'], axis = 1)
    maygames = maygames.rename(columns = 
                               {
        'PTS':'Away Points',
        'PTS.1':'Home Points'
    })

    #dump the data to reference for later
    pickle.dump(maygames,open(os.path.join(filepath,'maygames'),'wb'))
else:
    maygames = pd.read_pickle('maygames')
    
maygames
# -


# The dataframe we will be working with is all NBA games from the 2020-2021 season played in May. We have 173 games to work with -- a relatively strong sample size.

# Let's start here with taking a sample with the default parameters just to see what the raw function itself actually does:

# In[10]:


maygames.sample()


# The default is for this function to return a single value from the dataframe as the sample. Using the right parameters can give you exactly the sample you're looking for, but all parameters of this function are optional.

# ## How many samples?

# The first step to taking a sample from a population of data is to figure out exactly how much data you want to sample. This function has two different ways to specify this -- you can either use the parameters n or frac, but not both.<br>
# <br>
# ### n <br>
#  * This is a parameter that takes in an integer. It represents the numebr of items from the specified axis to return. If neither n or frac aren't specified, we are defaulted with n = 1.<br>
#  <br>
# ### frac<br>
#  * This is a parameter that takes in a float value. That float returns the fraction of data that the sample should be, representative of the whole population. Generally speaking, the frac parameter is usually between 0 and 1, but can be higher if you want a sample larger than the population<br>
#  <br>
# ### Clarification <br>
# It's important to note that if just any number is typed in, the sample function will think that it is taking an input for n.

# In[11]:


maygames.sample(n = 5)


# In[12]:


maygames.sample(frac = 0.5)


# In[13]:


print(len(maygames))
print(len(maygames.sample(frac = 0.5)))


# ## Weights and random_state

# The weights and random_state paramteres really define the way that we are going to sample from our original dataframe. Now that we have the parameter that tells us how many datapoints we want for our sample, it is imperative that we sample the right way. <br>
# <br>
# ### Weights<br>
# <br>
# Weights helps define the probabilities of each item being picked. If the parameter is left untouched, then the default for this is that all datapoints have an equal probability of being chosen. You can choose to specify the weights in a variety of ways. <br>
# <br>
# If a series is used as the parameter, the weights will align itself with the target object via the index.<br>
# <br>
# If a column name is used, the probabilities for being selected will be based on the value of that specific column. If the sum of the values in that column is not equal to 1, the weights of those values will be normalized so that they sum to 1. If values are missing, they will be treated as if they are weighted as 0. 

# In[14]:


maygames.sample(n = 10, weights = 'Attend.')


# The sample above took in 10 datapoints, and was weighted based on the game attendance, so that the games with more people at them had a higher chance of being picked. 

# ### Random_state<br>
# <br>
# Random state is essentially the parameter for the seed we want. This creates a sample that is reproducible if you want it to be. Generally, an integer is inputted for the parameter, but an np.random.RandomState object can be inserted if wanted. The default value for this is None.

# In[15]:


sample_1 = maygames.sample(n = 10, weights = 'Attend.', random_state = 1)
sample_1


# In[16]:


sample_2 = maygames.sample(n = 10, weights = 'Attend.', random_state = 1)
sample_2


# In[17]:


sample_1 == sample_2


# As you can see, the random_state parameter creates a sample that can be reproduced for future uses, which can prove to be incredibly helpful.

# ## Replace and ignore index

# The last few optional parameters we have are replace and ignore index. Both can be advantageous in their own right. <br>
# <br>
# ### Replace<br>
# <br>
# The parameter replace specifies whether we want to be able to sample with or without replacement. It takes in a Boolean as input. If True, then the datapoint has the ability to be chosen again into the sample. If False, the datapoint is removed from the pool of possible points to be chosen. 

# In[18]:


maygames.sample(
    n = 10,
    weights = 'Attend.',
    random_state = 1,
    replace = True)


# ### Ignore_index<br>
# <br>
# The ignore_index parameter is useful if you want your index to be relabeled instead of having the original index labels in the sample. This takes in a Boolean input. If True, the resulting index is relabeled, but if False (default), then the resulting index stays how it was. 

# maygames.sample(<br>
#     n = 10,<br>
#     weights = 'Attend.',<br>
#     random_state = 1,<br>
#     replace = True,<br>
#     ignore_index = True)<br>
# --

# ___<br>
# ## Idioms-if/then<br>
# **Name: Junqian Liu**<br>
# <br>
# UM-mail: junqianl@umich.edu

# In[21]:


import numpy as np
import pandas as pd


# ## Dataframe method
# - The dataframes allows us to change the values of one or more columns directly by the conditions
# - df.loc allows us to choose which columns to work as the condition, and which columns to be changed based on the conditions
# - More specifically, it works as df.loc[conditions, target columns] = values

# In[22]:


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

# In[23]:


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

# In[24]:


df3 = pd.DataFrame(
    {"apple": [4, 5, 6, 7], "boy": [10, 20, 30, 40], "cat": [100, 50, -30, -50],
    "dog": [3, 5, 0, 6]}
)
df3["elephant"] = np.where(df["apple"] > 5, "water", "banana")
df3

