#!/usr/bin/env python
# coding: utf-8

# In[184]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[185]:


bank=pd.read_csv(r"C:\Users\HP\Downloads\Python materials\Bank Data.csv")
bank.head(2)


# In[186]:


from sklearn.preprocessing import LabelEncoder
le_default=LabelEncoder()
bank['default_n']=le_default.fit_transform(bank['default'])


# In[187]:


from sklearn.preprocessing import LabelEncoder
le_default=LabelEncoder()
bank['deposit_n']=le_default.fit_transform(bank['deposit'])


# In[189]:


bank.drop('default', axis=1, inplace=True)


# In[190]:


bank.drop('deposit', axis=1, inplace=True)


# In[191]:


#bank1=bank.drop(['default','deposit'],axis='columns',inplace = True)
bank.head(2)


# In[192]:


#barplot
sns.barplot(x='default_n',y='job',data=bank)


# In[193]:


bank[bank.isnull().any(axis=1)].count()


# In[194]:


bank.describe()
#mean and median should be almost equal
#descriptive stats
#only possible on continuous and numeric data


# In[195]:


#Boxplot for 'age'
age=sns.boxplot(x=bank["age"])


# In[196]:


#Distribution of Age
sns.distplot(bank.age,bins=50)


# In[197]:


#Boxplot for 'duration'
sns.boxplot(x=bank["duration"])


# In[198]:


# Explore People who made a deposit Vs Job category
jobs = ['management','blue-collar','technician','admin.',
        'services','retired','self-employed','student',\
        'unemployed','entrepreneur','housemaid','unknown']

for j in jobs:
    print("{:20} : {:10}". format(j, 
                                 len(bank[(bank.deposit_n == 0) 
                                               & (bank.job ==j)])))


# In[199]:


# Different types of job categories and their counts
bank.job.value_counts()


# In[200]:


# Combine similar jobs into categiroes
bank['job'] = bank['job'].replace(['management', 'admin.'], 'white-collar')
bank['job'] = bank['job'].replace(['services','housemaid'], 'pink-collar')
bank['job'] = bank['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')


# In[201]:


bank['job']


# In[202]:


# New value counts
bank.job.value_counts()


# In[203]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[204]:


job=bank['job'] #bringing col separately
marital=bank['marital']
education=bank['education']


# In[205]:


job_n=le.fit_transform(job)
marital_n=le.fit_transform(marital)
education_n=le.fit_transform(education)


# In[206]:


from sklearn.preprocessing import OneHotEncoder
binary=OneHotEncoder(categories='auto') #3 separate cols for C,Q,S
cols1=binary.fit_transform(job_n.reshape(-1,1))
cols2=binary.fit_transform(marital_n.reshape(-1,1))
cols3=binary.fit_transform(education_n.reshape(-1,1))


# In[207]:


matrix1=cols1.toarray()
matrix1


# In[208]:


matrix2=cols2.toarray()
matrix3=cols3.toarray()


# In[209]:


job_df=pd.DataFrame(matrix1,columns=['white-collar','blue-collar','technician','other','pink-collar','self-employed','entrepreneur'])


# In[210]:


marital_df=pd.DataFrame(matrix2,columns=['married','single','divorced'])
education_df=pd.DataFrame(matrix3,columns=['secondary','tertiary','primary','unknown'])


# In[211]:


bank_df=pd.concat([bank,job_df,marital_df,education_df],axis=1) #added converted dummies with previous data
bank_df.head(2)


# In[212]:


bank_df.drop('job', axis=1, inplace=True)


# In[213]:


bank_df.drop('marital', axis=1, inplace=True)


# In[214]:


bank_df.drop('education', axis=1, inplace=True)


# In[215]:


bank_df.head(2)


# In[216]:


bank_df.poutcome.value_counts()


# In[217]:


# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_df['poutcome'] = bank_df['poutcome'].replace(['other'] , 'unknown')
bank_df.poutcome.value_counts()


# In[218]:


# Drop 'contact', as every participant has been contacted. 
bank_df.drop('contact', axis=1, inplace=True)


# In[220]:


bank_df.head(2)


# In[221]:


# values for "default" : yes/no
#bank["default"]
#bank['default_cat'] = bank['default'].map({'yes':1, 'no':0})
#bank.drop('default', axis=1,inplace = True)


# In[222]:


# values for "default" : yes/no
#bank_df['default_n'] = bank_df['default'].map({'yes':1, 'no':0})
#bank_df.drop('default', axis=1,inplace = True)


# In[223]:


# values for "housing" : yes/no
bank_df["housing_cat"]=bank_df['housing'].map({'yes':1, 'no':0})
bank_df.drop('housing', axis=1,inplace = True)


# In[224]:


# values for "loan" : yes/no
bank_df["loan_cat"] = bank_df['loan'].map({'yes':1, 'no':0})
bank_df.drop('loan', axis=1, inplace=True)


# In[225]:


# day  : last contact day of the month
# month: last contact month of year
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_df.drop('month', axis=1, inplace=True)
bank_df.drop('day', axis=1, inplace=True)


# In[226]:


bank_df.head(2)


# In[238]:


# pdays: number of days that passed by after the client was last contacted from a previous campaign
#-1 means client was not previously contacted

print("Customers that have not been contacted before:", len(bank_df[bank_df.pdays==-1]))
print("Maximum value of pdays:", bank_df['pdays'].max())


# In[240]:


# Map padys=-1 into a large value (10000 is used) to indicate that 
#it is so far in the past that it has no effect
bank_df.loc[bank_df['pdays'] == -1, 'pdays'] = 10000


# In[241]:


# Create a new column: recent_pdays 
bank_df['recent_pdays'] = np.where(bank_df['pdays'], 1/bank_df.pdays, 1/bank_df.pdays)

# Drop 'pdays'
bank_df.drop('pdays', axis=1, inplace = True)


# In[242]:


bank_df['recent_pdays']


# In[228]:


#bank_df= pd.get_dummies(data=bank_df, columns = ['job', 'marital', 'education', 'poutcome'], \
                                   #prefix = ['job', 'marital', 'education', 'poutcome'])


# In[229]:


# Scatterplot showing age and balance
bank_df.plot(kind='scatter', x='age', y='balance')
# Across all ages, majority of people have savings of less than 20000.


# In[231]:


#bank_with_dummies.plot(kind='hist', x='age', y='duration')


# In[232]:


#barplot
#sns.barplot(x='education',y='default_cat',data=bank)


# In[233]:


#barplot
#sns.barplot(x='job',y='deposit_cat',data=bank)


# In[234]:


#barplot
#sns.barplot(x='job',y='default_cat',data=bank)


# In[235]:


#barplot
#sns.barplot(x='marital',y='default_cat',data=bank)


# In[236]:


#barplot
#sns.barplot(x='marital',y='deposit_cat',data=bank)


# In[243]:


# People signed up to a term deposite having a personal loan (loan_cat) and housing loan (housing_cat)
len(bank_df[(bank_df.deposit_n == 1) &
                      (bank_df.loan_cat) & (bank_df.housing_cat)])


# In[244]:


# People signed up to a term deposite with a credit default 
len(bank_df[(bank_df.deposit_n == 1) & (bank_df.default_n ==1)])


# In[245]:


len(bank[bank.education=='tertiary'])


# In[246]:


len(bank[bank.education=='primary'])


# In[247]:


# Bar chart of "previous outcome" Vs "call duration"
plt.figure(figsize = (6,4))
sns.barplot(x='poutcome', y = 'balance', data = bank_df)


# In[248]:


# The Correltion matrix
corr = bank_df.corr()
corr


# In[249]:


# Heatmap
plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, 
            cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')


# In[251]:


# Extract the deposte_cat column (the dependent variable)
corr_deposite = pd.DataFrame(corr['deposit_n'].drop('deposit_n'))
corr_deposite.sort_values(by = 'deposit_n', ascending = False)


# In[265]:


x = bank_df.drop('deposit_n', 1)
y= bank_df.deposit_n
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)


# In[263]:


bank_df.head(2)


# In[264]:


bank_df.drop('poutcome', axis=1, inplace=True)


# In[266]:


x_train.shape


# In[267]:


y_train.shape


# In[268]:


x_test.shape


# In[269]:


y_test.shape


# In[270]:


# Decision tree with depth = 2
dt2 = tree.DecisionTreeClassifier(random_state=1,max_depth=2)
dt2.fit(x_train, y_train)


# In[271]:


dt2_score_train = dt2.score(x_train, y_train)
print("Training score: ",dt2_score_train)
dt2_score_test = dt2.score(x_test, y_test)
print("Testing score: ",dt2_score_test)


# In[272]:


# Decision tree with depth = 3
dt3 = tree.DecisionTreeClassifier(random_state=1, max_depth=3)
dt3.fit(x_train, y_train)


# In[273]:


dt3_score_train = dt3.score(x_train, y_train)
print("Training score: ",dt3_score_train)
dt3_score_test = dt3.score(x_test, y_test)
print("Testing score: ",dt3_score_test)


# In[275]:


# Decision tree with depth = 4
dt4 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)
dt4.fit(x_train, y_train)
dt4_score_train = dt4.score(x_train, y_train)
print("Training score: ",dt4_score_train)
dt4_score_test = dt4.score(x_test, y_test)
print("Testing score: ",dt4_score_test)


# In[276]:


# Decision tree with depth = 6
dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)
dt6.fit(x_train, y_train)
dt6_score_train = dt6.score(x_train, y_train)
print("Training score: ",dt6_score_train)
dt6_score_test = dt6.score(x_test, y_test)
print("Testing score: ",dt6_score_test)


# In[277]:


# Decision tree: To the full depth
dt1 = tree.DecisionTreeClassifier()
dt1.fit(x_train, y_train)
dt1_score_train = dt1.score(x_train, y_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt1.score(x_test,y_test)
print("Testing score: ", dt1_score_test)


# In[278]:


print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))
print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))
print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))
print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))
print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))
print('{:1} {:>25} {:>20}'.format(6, dt6_score_train, dt6_score_test))
print('{:1} {:>23} {:>20}'.format("max", dt1_score_train, dt1_score_test))


# In[ ]:


#It could be seen that, higher the depth, training score increases and matches perfects with the training data set. However higher the depth the tree goes, it overfit to the training data set. So it's no use keep increasing the tree depth. According to above observations, tree with a depth of 2 seems more reasonable as both training and test scores are reasonably high.

