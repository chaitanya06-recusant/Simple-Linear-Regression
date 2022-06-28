#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from ml_metrics import rmse


# In[2]:


data = pd.read_csv('delivery_time (1).csv')


# In[3]:


data


# # Analysing the data using info and describe function

# In[4]:


data.info()


# In[5]:


data. describe()


# # Checking for linear association and correlation using scatter plot and reg plot

# In[8]:


data = data.rename({'Delivery Time': 'DT','Sorting Time': 'ST'}, axis=1)


# In[10]:


plt.plot(data.DT, data.ST,'bo')


# In[11]:


sns.regplot('DT','ST', data)


# In[12]:


data.corr()


# Hence, the above scatter and regression plot show a positive linear association and the two variables have a correlation of 0.825 > 0.8. So, they have a strong correlation.

# # checking for outliers using box plot and histogram

# In[15]:


plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
data['DT'].hist()
plt.subplot(1,2,2)
data.boxplot(column=['DT'])

plt.show()


# In[16]:


plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
data['ST'].hist()
plt.subplot(1,2,2)
data.boxplot(column=['ST'])

plt.show()


# From the above histogrms and boxplots, we know that there are no outleirs present inside the DT(Delivery Time) and ST (Sorting Time) variable.

# # Trying to fit the model without transformation

# In[18]:


model = smf.ols('DT~ST', data = data).fit()


# In[19]:


model.summary()


# In[20]:


model.params


# In[21]:


model.pvalues


# In[22]:


model.rsquared


# In[23]:


model.rsquared_adj


# The above results show that, the R squared values is 0.682 which means the model is only 68% accurate. Although the p-value = 0.000004 which is less than 0.05. Hence ST is a significant variable.

# In[24]:


pred = model.predict(data.ST)


# In[25]:


pred


# In[29]:


actual = data.DT


# In[30]:


rmse(pred,actual)


# # Model with logarithmic transformation

# In[32]:


model1 = smf.ols('DT~np.log(ST)',data).fit()


# In[33]:


model1.summary()


# Here as per above result in this case R-squared value is 0.695 which is greater than our model1 but not greater than 0.85. We can say model2 is better than model1 . But not the best fit model to predict Delivery_time

# In[34]:


pred1 = model1.predict(data.ST)
pred1


# In[35]:


rmse(pred1, actual)


# 
# # As in model1 the R-squared value is not also good. So we need to do another transformation to get better R-squared value.
# # Applying Exponential transformation and predict a new model

# In[36]:


model2 = smf.ols('DT~np.exp(ST)', data).fit()


# In[37]:


model2.summary()


# Here as per above result in this case R-squared value is 0.361 which is lesser than 0.85. We cannot take this model to predict Delivery_time

# # Model 3 using reciprocal transformation

# In[40]:


model3 = smf.ols('DT~np.reciprocal(ST)', data).fit()


# In[41]:


model3.summary()


# # Model 4 using square root transformation

# In[42]:


model4 = smf.ols('DT~np.sqrt(ST)', data).fit()


# In[44]:


model4.summary()


# Here as per above result in this case R-squared value is 0.696 which is lesser than 0.85. We cannot take this model to predict Delivery_time

# In[45]:


pred4 = model4.predict(data.ST)


# In[46]:


rmse(pred4, actual)


# # Model 5 using exponential function on target variable

# In[47]:


model5 = smf.ols('np.log(DT)~ST', data).fit()


# In[48]:


model5.summary()


# # Here as per above result in this case R-squared value is 0.711.
# # p- values is less than 0.05, it is significant
# # Checking RMSE value and predict delivery time

# In[49]:


pred5 = np.exp(model5.predict(data.ST))


# In[50]:


pred5


# In[51]:


rmse(pred5,actual)


# # Conclusion - Comparing between all models , model5 has higher R-squared value i.e. 0.711 as comapare to others.
# # From the above data we know higher R-squred value and lower RMSE value gives better model.
# # Hence the model5 is better model to predict delivery_time

# In[ ]:




