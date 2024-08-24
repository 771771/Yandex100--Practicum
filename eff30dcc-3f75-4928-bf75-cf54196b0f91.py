#!/usr/bin/env python
# coding: utf-8

# ----------
# ## Review
# 
# Hi, my name is Daria! I'm reviewing your project. 
# 
# You can find my comments under the heading «Review». 
# I’m using __<font color='green'>green</font>__ color if everything is done perfectly. Recommendations and remarks are highlighted in __<font color='blue'>blue</font>__. 
# If the topic requires some extra work, the color will be  __<font color='red'>red</font>__. 
# 
# You did an outstanding job! Every step was correct and well commented. Congratulations on your first completed project! Good luck in future learning :)
# 
# ---------
# 

# ## Analyzing borrowers’ risk of defaulting
# 
# Your project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Your report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.

# ### Step 1. Open the data file and have a look at the general information. 

# In[ ]:





# In[1]:


#Importing relevant libraries
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer 
english_stemmer = SnowballStemmer('english')


# In[2]:


#read the csv
credit_scoring_eng = pd.read_csv('/datasets/credit_scoring_eng.csv') 


# In[3]:


credit_scoring_eng.columns = credit_scoring_eng.columns.str.strip()


# In[4]:


#overall info
#days employed and total income must have null values and is in float type,
# the reason I will take care of the days employed is to further understand what we can take out of 
#this data. If we want to extract conclusion from total income we have to fill in the missing values
#I would want for both of them to be int because in some cases it can be harder to manipulate floats.


credit_scoring_eng.info()


# In[5]:


#general statistic information
credit_scoring_eng.describe()


# In[6]:


#notice the head of the file
credit_scoring_eng.head()


# In[7]:


credit_scoring_eng.tail()


# In[8]:


#I see taht there are 76 indiviiduals with 20 children(!) can be true will further investigate
#about -1 children seems like a typo 
credit_scoring_eng['children'].value_counts()


# ----------
# <font color='green'>
# 
# ## Review
# 
# Good point!
#     
# </font>
# 
# ---------

# In[9]:


credit_scoring_eng['days_employed'].value_counts()
# there are minus values which don't make sense, could be that its days that they are not employed
# i will futher investigate, it does have astronomical values aswell


# In[10]:


credit_scoring_eng['dob_years'].value_counts()
# 101 people 0 years old? have to further look at it, 
#maybe that's how they represent that they have passed away
#I think it's a wrong input anyway


# In[11]:


credit_scoring_eng['education'].value_counts()

print(credit_scoring_eng['education'].str.lower().value_counts())

# after lowering, we get only 5 types of degrees, seems like there are very few graduate degrees
# will be hard to get conclusions we can count on from it 

# I will speculate that the higher the education the higher the income will be, 
# and if it was a real question here I would like to investigate it.


# In[12]:


credit_scoring_eng['education_id'].value_counts()
# just a number representation of the education


# In[13]:


credit_scoring_eng['family_status'].value_counts()

# The data looks good, the number of widows and divorced are alot lower than the rest
# might think twice about the conclusion, and the married are very (!) high number,
# I would suspect that if we see some kind of data from it we will tend to believe it.


# In[14]:


#Interestingly found out that there is one person in the table that doesn't identify as a certain gender
#I think it should remain that way
credit_scoring_eng['gender'].value_counts()
# if we wouldve manipulated that data i would've needed to delete this row


# In[15]:


#are some values in the income type which are very rare- could've been an error in the input,
credit_scoring_eng['income_type'].value_counts()

##
#credit_scoring_eng.loc[credit_scoring_eng['income_type']=='unemployed']
#credit_scoring_eng.loc[credit_scoring_eng['income_type']=='entrepreneur']
#credit_scoring_eng.loc[credit_scoring_eng['income_type']=='paternity / maternity leave']
#credit_scoring_eng.loc[credit_scoring_eng['income_type']=='student']
##
#unemployed:
#looks odd because they do have total income
#entreperenuer:
#has total income aswell for one and the second is NaN, sounds very shady
#paternity:
#Has also total income aswell
#student:
#has total income aswell

#conclusion from this, we can't have conclusion from this data because the number of individuals 
#is so low, aka. we can't have any conclusion about them because of the low dataset,
#I would have just add the student to any of the groups, or leave them that way because
# we won't talk about this further.


# In[16]:


#as an overview I see most of the people have paid their loans.
credit_scoring_eng['debt'].value_counts()


# In[17]:


#many purposes have to catagorize them later.
#might have been because every person in the bank wrote (without a pre made options) their 
#purpose - resulting with many purposes covering a small amount of acutal purposes.


credit_scoring_eng['purpose'].value_counts()


# 

# There are many factors at play in order to evaluate the ability of a potential borrower to repay their loan, the purpose, income, imployment all have to be evaluated, 
# 
# There are  missing values in days and income
# 
# many purposes that have to be catagorized
# 
# There might be correlation between the level of education and the income and ability 
# to pay the loan, we aren't focusing on that now so just a point to notice
# 

# ----------
# <font color='green'>
# 
# ## Review
# 
# Great data study! You were able to identify lots of errors in our dataset, let's see what we can do with them :)
#     
# </font>
# 
# ---------

# 

# ### Processing missing values

# In[18]:


#change column names for more intuitive undertanding of the chart
credit_scoring_eng = credit_scoring_eng.rename(columns={"dob_years": "age", "children" : "children_number"})

credit_scoring_eng.head()


# In[19]:


#because I have empty cells in the days employed meaning that they could on one hand not be working or its a typo.
credit_scoring_eng['days_employed'] = credit_scoring_eng['days_employed'].fillna(value = 0)


#there are 10% of the values empty and I think it is on purpose,
#because they are unimployed,although some of them have income and their 


percentage_null_days = (pd.isna(credit_scoring_eng['days_employed']).sum())/ credit_scoring_eng.shape[0]

print(percentage_null_days)

#Some of the values are negative in the days employed column although the total income is positive which doesnt make sense that's why I switching them to positive
credit_scoring_eng['days_employed'] = (credit_scoring_eng['days_employed'].abs())


#Adding a new column in order to understand if the days employed makes sense, seems like this is distorted with massive numbers
def days_to_years (days):
    years = days/365
    return years


credit_scoring_eng['years_employed'] = credit_scoring_eng['days_employed'].apply(days_to_years)



credit_scoring_eng.loc[(credit_scoring_eng['years_employed'] > credit_scoring_eng['age'])]

sorted_years = credit_scoring_eng.sort_values(['years_employed'],ascending=False)
#sorted_years.tail(5)
# here we see that the years employed it too high (a thousand years!) and we should see a relevant category


#after switching to years employed i see that the column is very corrupted I decide to now
#stop investigating it.

credit_scoring_eng = credit_scoring_eng.drop(columns=['days_employed']) # for style reasons 
#i will keep only years employed


# ----------
# <font color='green'>
# 
# ## Review
# 
# OK, filling with zeros is good when you do think that these clients haven't worked a day. Otherwise I'd suggest  you to fill missing values with some unrealistic value (e.g. -1), just to be able to exclude them from further analysis.
#     
# </font>
# 
# ---------

# In[20]:


#there is the exact same empty values in total income, probably because they are related,
#but I will fill them with median in order so it won't change the conclusion drastically
#if I insert 0 there and won't be affected by the high income individuals (compared to if I insert mean)

income_median = credit_scoring_eng['total_income'].median()

credit_scoring_eng['total_income'] =credit_scoring_eng['total_income'].fillna(income_median)


# ----------
# <font color='green'>
# 
# ## Review
# 
# A reasonable decision :)
#     
# </font>
# 
# ---------

# In[21]:


#(I) number of children - there are empty values- will switch with 0 - speculated to be because of human error
# days employed- switched to years employed, has negative values probably due to mistake, and have odd values of 1000 years, 
#will not regard this column
#age - odd values of 0 will be disgarded
credit_scoring_eng['children_number'] = credit_scoring_eng['children_number'].fillna(value = 0)

print (credit_scoring_eng['children_number'].isnull().sum())


# In[22]:


#now if there are empty age values they are filled up, in order not to lose the whole row
credit_scoring_eng['age'] = pd.to_numeric(credit_scoring_eng['age'],errors='coerce')



# In[23]:


#here we can see that there are 22 individuals that are younger than 35 with 20 children, doesn't make sense- should drop them

invalid_children = credit_scoring_eng.loc[(credit_scoring_eng['children_number'] == 20 )  & (credit_scoring_eng['age'] < 35)]

invalid_children.info()


#and are 52 older than 35 with 20 children, which physically makes sense.

valid_children = credit_scoring_eng.loc[(credit_scoring_eng['children_number'] == 20 )  & (credit_scoring_eng['age'] > 35)]


#I assume that the 22 individuals younger than 35 that have 20 kids is a typo and means 2 kids instead.
credit_scoring_eng.loc[(credit_scoring_eng['children_number'] == 20 )  & (credit_scoring_eng['age'] < 35), 'children_number'] = 2
# the rest of the people that are over 35 I will leave unchanged beacuse it is physically possible that they have 2- children


credit_scoring_eng.loc[credit_scoring_eng['children_number'] == -1, 'children_number'] = 1
#there are 47 ,
#-1 children seems the most of the individuals are married and I think there've been a typo there


print(credit_scoring_eng['children_number'].value_counts()) 
# Great! no more minus and changed values from 20 children.



# #missing values in total income, days employed and children
# the missing values types that were detected are- 
# 
# total_income- quantitative
# days employed quantitative aswell and is correlated with total_income, 
# children is quantitative
# 
# 
# #I assume the children section is empty because the person has no children, i have filled those with value = 0 - — Missing not at random MNAR
# 
# #the total income and days employed are missing due to riterment, owning a buisness or civil servant, left them as null
# 
# 
# 
# #there are 2174 (10%) empty total_income lines, I have added the values of the mean there because it is only 10 percent of the table.
# 
# #I changed the years employed to absolute values because I see that people do have total income even when the employment is negative meaning that its an error
# 
# 
# #there are 158284 (73% of the chart) of the rows have people that started working from age 13. the others have #values that aren't valid in my opinion
# 
# #there are values in age column that are zero, we are not discussing ages so we will skip that
# #there are empty values in age which are human input error
# 

# ----------
# <font color='green'>
# 
# ## Review
# 
# Great work on data processing! :)
#     
# </font>
# 
# ---------

# 

# ### Data type replacement

# In[24]:


#Switch data type of income to int to categorize in a later stage

#switching years employed for neater data

try:

    credit_scoring_eng['total_income'] = credit_scoring_eng['total_income'].astype('int')  
#because I have total income as float I will use astype to change them all to int   


    credit_scoring_eng['years_employed'] = credit_scoring_eng['years_employed'].astype('int') 
# because I have years employed with empty cells I switched them to 0 and then I can change 

except:
    print("There are strings in income and years, check it out")
    
# if we have strings it would make our code crash added this try except.


# ### Conclusion

# The total income is now integer, while the empty cells were filled up with median in order to get the most accurate results. 
# * I decided to use median and not mean because median is not affected with drastically high/low values.

# ### Processing duplicates

# In[25]:


#there are sporadic data types all have to be in lower case in order to further catagorize
try:
    credit_scoring_eng['education'] = credit_scoring_eng['education'].str.lower()

#credit_scoring_eng.duplicated().sum() # 71 duplicated rows

#credit_scoring_eng.loc[credit_scoring_eng.duplicated(), :]   # gives rowif a row is at least twice
#keeping the first one that was found, it is the default, I could've done 'last' to change it

except:
    print("There's string in the education column ") # if we have string there accidentaly the code
    #will crash
    
    
credit_scoring_eng=credit_scoring_eng.drop_duplicates().reset_index(drop=True) 

# Used this method because i got rid of duplicates in all of the dataset



# ### Conclusion

# deleted duplicates in the chart after I made the education lower, because they were many values
# repeating themselves
# 
# I used drop duplicates on all of the table as a whole
# if there are duplicates that i've missed that have to be retouched.
# 
# 
# 
# #the duplicates might have appeared because:
# #they were merged from different datasets
# #they were inserted by different people without knowing the prior input
# #they were inserted at different times and the person which inserted the data didn't remember the past input.

# ----------
# <font color='green'>
# 
# ## Review
# 
# All good :)
#     
# </font>
# 
# ---------

# ### Categorizing Data

# In[26]:


print(credit_scoring_eng['purpose'].str.lower().value_counts())
credit_scoring_eng['purpose'] = credit_scoring_eng['purpose'].str.lower()


# In[27]:


#Categorize the Data into four groups and print out to check that I have gotten only them
# I am catagorizing the data with the lemmatizer because it will help me identify correctly
#all of the groups of purposes with a larger context and not just find the word with the
# loop "is in" purpose.

# the concept is that we want to be able to find all of the occuring words in that group to analyze all of the customers
#seperated to main purpose groups.


import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemma = WordNetLemmatizer()


purpose_words = [
   ['house', 'property', 'real', 'estate', 'resident', 'housing'],
   ['car', 'cars'],
   ['education', 'educated', 'university' ],
   ['wedding', 'wedding', 'ceremony']
]

#group by purpose and tokenize them by group names
   
def grouping_purpose(row):

   # function will return the words in each purpose and check it in wordnet database
   
   whole_purpose = row['purpose'] # for each row in purpose
   
      
   words = nltk.word_tokenize(whole_purpose) # purpose tokenized
   
   lemmas = [wordnet_lemma.lemmatize(w, pos = 'n') for w in words] # lemmatize the words, crates a list by default
   
   for w in lemmas: # for word in list created by lemmas
                   
           for r in purpose_words: #check if word is in purpose words
               
               if w in r: # compare words with key words
                   
                   return r[0] # name the group
               
               
# add a new column with into dataset
credit_scoring_eng['purpose_group'] = credit_scoring_eng.apply(grouping_purpose, axis = 1)
      

credit_scoring_eng['purpose_group'].value_counts()


# ----------
# <font color='green'>
# 
# ## Review
# 
# Good, I agree with your choice of categories :) 
#     
# </font><font color='blue'>
# 
# I think, stemming might be more suitable for such task, as it cuts your words significantly. Probably "cars" and "car", "house" and "housing" etc would be the same word after stemming.
#     
# </font>
# 
# ---------

# In[28]:


def sort_income(data):
    if ((data > 23203) & (data <= 31286)):
        data ='medium-high'
    elif ((data <= 23203) & (data > 17247)):
        data = 'medium-low'
    elif (data <= 17247):
        data =  'low'
    else:
        data =  'high'
    return data

credit_scoring_eng['income_group'] = credit_scoring_eng['total_income'].apply(sort_income)
credit_scoring_eng.head(3)



# In[29]:


marital_debt = credit_scoring_eng.groupby('family_status').debt.agg(['sum','count'])

marital_debt['percentage'] = (marital_debt['sum'] / marital_debt['count'])* 100

marital_debt = marital_debt.drop(columns=['count', 'sum'])


marital_debt



# In[30]:


credit_scoring_eng.groupby('debt').children_number.mean()


# In[31]:


group_debt = credit_scoring_eng.groupby('purpose_group').debt.agg(['sum','count'])

group_debt['percentage'] = (group_debt['sum'] / group_debt['count']) * 100

group_debt = group_debt.drop(columns=['count', 'sum'])

group_debt


# In[32]:


children_debt = credit_scoring_eng.groupby('children_number').debt.agg(['sum','count'])

children_debt.loc["Has Children"] = children_debt[1:7].sum() # sum in  order to break to two values

# 

#retouching to get a more neat chart

children_debt = children_debt.rename( index={0: "No Children"})

children_debt['default percentage'] = (children_debt['sum'] / children_debt['count']) * 100

children_debt = children_debt.drop(columns=['count', 'sum'])

children_debt = children_debt.drop(children_debt.index[1:7])

#

print( "The pecentage of defaulted loans if the customer has or doesn't have children:")
children_debt


# In[33]:


income_debt = credit_scoring_eng.groupby('income_group').debt.agg(['sum','count'])


income_debt['percentage'] = (income_debt['sum'] /income_debt['count']) * 100

income_debt = income_debt.drop(columns=['count', 'sum'])
              
               
income_debt


# In[38]:


lambda x: x.sum()/x.count()*100


# ----------
# <font color='green'>
# 
# ## Review
# 
# That's perfect, you correctly choose the metric to base your conclusion on :)
#     
# </font><font color='blue'>
# 
# A little hint on ``.agg``: you could've add a lambda-expression to list of functions to calculate ``percentage`` automatically (``lambda x: x.sum()/x.count()*100``)
#     
# </font>
# 
# ---------

# ### Conclusion

# 

# ### Step 3. Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

# In[34]:


#There is a relation between having kids and repaying the loan on time, 
#because there is a 2.5 percent (!) favoring the No children group.

#if we break down the has children, we can see that they are around 9 and for 5 children, that 
#didn't default - isn't a reliable value because of the low customer count in that group, isn't
#representing

#so overall we see that we have a bigger customer count in the non children group and still have
#a lower default rate gives me the feeling that it's a more reliable customer.


# ### Conclusion

# 

# - Is there a relation between marital status and repaying a loan on time?

# In[35]:


# i think there is relation to between martial status and paying debt
#widow:
# we should think twice about the widower group that seems to be significantly better customer,
# the amount of widowers is very low in comparison to married status, but we can see a glimpse of
# informatio here that the amout of defaults is very low even at this low customer amount

#divorced
# similarly to the widows group -has a low amount of customers.

#married:
#biggest amout of customers, around 12 thousand seems like they are better customers than
#unmarried and civil partnership that are around 10% 

#I would want a larger amount of customers in order to adress in more accurately

# So because the married group has more depth and related to other groups it has good debt paying
# ability i'd say that it's the most reliable group.


# ### Conclusion

# 

# - Is there a relation between income level and repaying a loan on time?

# In[36]:


# as we can speculate before looking at the data the group with the highest income is going to have the least default
#we find out that it is true here, and percentage of deafult is the lowest, 

#on a side note we can see taht the lowest income group has better debt returns than the medium high and medium low mabybe
#because they ask for smaller loans, the meium high is the most dangerous group to trust out of this informtion table.


# ### Conclusion

# 

# - How do different loan purposes affect on-time repayment of the loan?

# In[37]:


# about the loan purpose we see that the lowest default percentage is onthe house groups, I may speculate its because that 
# they know they are in a long term loan and need to be more stable, meaning that they got a stable job in order to be able to 
# bring back the loan

#this conclusion is most accurate from all of the other groups under inspection because there is the highest amount of 
#customers in that group, making it significantly better.


# ### Conclusion

# 

# ### Step 4. General conclusion

# The most reliable customers are married , without children, above the 75 percent of the median income that the purpose of their loan is house related.

# ----------
# <font color='green'>
# 
# ## Review
# 
# All your conclusions are correct and consistent with observations :) Great work!
#     
# </font>
# 
# ---------

# ### Project Readiness Checklis
# 
# Put 'x' in the completed points. Then press Shift + Enter.

# - [x]  file open;
# - [x]  file examined;
# - [x]  missing values defined;
# - [x]  missing values are filled;
# - [x]  an explanation of which missing value types were detected;
# - [x]  explanation for the possible causes of missing values;
# - [X]  an explanation of how the blanks are filled;
# - [X]  replaced the real data type with an integer;
# - [X]  an explanation of which method is used to change the data type and why;
# - [X]  duplicates deleted;
# - [X]  an explanation of which method is used to find and remove duplicates;
# - [X]  description of the possible reasons for the appearance of duplicates in the data;
# - [X]  data is categorized;
# - [X]  an explanation of the principle of data categorization;
# - [X]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [X]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [X]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [X]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [X]  conclusions are present on each stage;
# - [X]  a general conclusion is made.

# In[ ]:




