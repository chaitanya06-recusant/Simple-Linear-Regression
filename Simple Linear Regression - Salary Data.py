#!/usr/bin/env python
# coding: utf-8

# # Salary_hike -> Build a prediction model for Salary_hike

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


salary = pd.read_csv('Salary_Data (1).csv')


# # Analysing the Salary Data using info and describe function

# In[3]:


salary.info()


# In[4]:


salary.describe()


# # checking the correlation between the two variables

# In[5]:


salary. corr()


# Since the |r|>0.8, Both the variables have a strong correlation

# # Checking the linear association using scatter plot

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


plt.plot(salary.YearsExperience, salary.Salary, 'bo')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# In[8]:


sns.regplot('YearsExperience', 'Salary', salary)


# Hence both the variable have a positive linear association with a strong correlation of 0.978242

# In[9]:


salary[salary.duplicated(keep = False)]


# There are no duplicate values

# # Checking for outliers

# In[10]:


plt.boxplot(salary.YearsExperience)


# In[11]:


plt.boxplot(salary.Salary)


# The above two boxplots clearly shows there are no outliers

# # Predict a model without applying transformation

# In[12]:


import statsmodels.formula.api as smf
model = smf.ols('Salary~YearsExperience', salary).fit()


# In[13]:


model.summary()


# In[14]:


model.params


# In[15]:


print(model.rsquared)


# Since, rsquared value is 0.95, it means the model is 95% accurate

# In[16]:


model.pvalues


# Since the p value<0.05, it means that the variable years of experience is a significant variable

# # Checking the rmse value

# In[17]:


pred = model.predict(salary.YearsExperience)
pred


# In[18]:


actual = salary['Salary']
actual


# In[19]:


from ml_metrics import rmse
rmse(pred,actual)


# As per above OLS Regression Results dependent variable is salary and here p value is less than 0.05 hence this is significant model
# Here R-squred value is 0.957 > 0.8. Hence we can say our model is good for Salary_hike

# 
# 
# # Checking  the model using transformation.

# In[21]:


model1 = smf.ols('Salary~np.log(YearsExperience)', data = salary).fit()


# In[22]:


model1.summary()


# Comparing the models with and without transformation, it is clear that   after applying logarithmic transformation on YearsExperience varibale, that R-squared value is 0.854 and p value is less than 0.05.
# 
# Conclusion, model without transformation has higher R-squared value i.e. 0.957 as comapare to model with transformation.
# 
# Hence the first model is better to predict Salary_hike

# In[ ]:




