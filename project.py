#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:50:51 2017

@author: Pan
"""
#Kia Duan(wd3fg)
#Huitong (Jo) Pan (hp4zw)
#Teresa Sun (js6sm)

#work cited1:https://www.kaggle.com/jkokatjuhha/we-are-from-our-childhood/notebook
#work cited2:https://www.kaggle.com/mikesch/who-are-the-money-savers
#work cited3:https://stackoverflow.com/questions/5306756/how-to-show-percentage-in-python
#work cited4:https://seaborn.pydata.org/generated/seaborn.countplot.html

#%% 
# -------- import data & libraries ---------
#matplotlib inline
import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import unittest
#import corrplot

# styling
pd.set_option('display.max_columns',150)
plt.style.use('bmh')
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

#read data
df1 = pd.read_csv('/Users/Pan/Google Drive/Data Science/CS 5010/project/responses.csv')
responses_pre = pd.read_csv('/Users/Pan/Google Drive/Data Science/CS 5010/project/responses.csv')
columns = pd.read_csv('/Users/Pan/Google Drive/Data Science/CS 5010/project/columns.csv')

# --------- data clieaning -------------------
#data cleanning to drop na
responses=responses_pre.dropna()
pd.isnull(responses).sum() > 0  #test to see if all columns are without na.

#brief overview of the data
responses.describe()
#         Loneliness  
#count    674.000000
#mean     2.890208             
#std      1.148599            
 

#%%
 
# Query 1 -------- loneliness by gender -------------
# subset the data into male group and female group
male = responses[responses.Gender == 'male'] 
male
female = responses[responses.Gender == 'female']
female

from scipy.stats import t, norm # library used for confidence interval

## build a function to calculate 95% confidence interval
def CI(x):
    mean = x.mean()
    std = x.std(ddof = 1)
    n = len(x)
    zstar = norm.ppf(0.975)
    CI_Lower = mean - zstar*std/(n**(1/2))
    CI_Upper = mean + zstar*std/(n**(1/2))
    
    return [CI_Lower, CI_Upper]
    
## build a function to calculate 95% confidence interval for difference between two groups   
def twosampCI(x,y):
   xdiff = x.mean() - y.mean()
   x_s = x.std(ddof=1)
   y_s = y.std(ddof=1)
   std_error = (((x_s**2)/len(x)) + ((y_s**2)/len(y)))**0.5
   zstar = norm.ppf(0.975)
   CI_Lower = xdiff - zstar*std_error
   CI_Upper = xdiff + zstar*std_error
   
   return [CI_Lower, CI_Upper]

# confidence interval for mean level of loneliness for young males
CI(male.Loneliness)
# [2.6256909561682722, 2.8963678673611399]

# confidence interval for mean level of loneliness for young females
CI(female.Loneliness)
# [2.8653563472744881, 3.0898675333225269]

# # confidence interval for difference in mean level of loneliness between young males and young females
twosampCI(male.Loneliness, female.Loneliness)
# [-0.39241715461297333, -0.040747902454629531]

# -----  count plot distribution of loneliness by gender -------
# male
var_of_interest = 'Loneliness'
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(male[var_of_interest], orient = 'h')
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)

# female
var_of_interest = 'Loneliness'
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(female[var_of_interest], orient = 'h')
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)

#%%
# Query 2 --------------- loneliness by whether is only child -------------------------
# subset the data into who is only child and who have siblings
onlychild = responses[responses['Only child'] == 'yes'] 
morechild = responses[responses['Only child'] == 'no'] 

# confidence interval for mean level of loneliness for only child
CI(onlychild.Loneliness)
# [2.7161452419176282, 3.1032095967920492]

# confidence interval for mean level of loneliness for people who have siblings
CI(morechild.Loneliness)
# [2.7876452273807413, 2.9811408997868885]

# confidence interval for difference in mean level of loneliness between single child and people who have siblings
twosampCI(onlychild.Loneliness, morechild.Loneliness)
# [-0.19108303436336005, 0.24165174590540772]

# -----  count plot distribution of loneliness by child -------

# only child
var_of_interest = 'Loneliness'
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(onlychild[var_of_interest], orient = 'h')
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)

# more child
var_of_interest = 'Loneliness'
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(morechild[var_of_interest], orient = 'h')
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)



#%%
from scipy import stats 
#-----------------------hypothesis testing-----------------------------
def check_values_in_variable(variable):
    uniquevalues= responses[variable].unique()
    return uniquevalues
#Check the unique values in a variable, return the values as a list

def divide_groups(groupingvar):
    uniquevalues = check_values_in_variable(groupingvar)
    groups=[]
    for value in uniquevalues:
        groups.append(responses.groupby([groupingvar]).get_group(value))
    return groups
#Divide the dataset by the grouping variable, using the unique values list
#returned by the above function       
        
def hypothesis_test(groupingvar,interestvar):
    uniquevalues = check_values_in_variable(groupingvar) #get unique values
    groups=[]
    groupnames=[]
    varlists=[]
    results=[]
    for value in uniquevalues: #append all the unique values to a groupname list
        groups.append(responses.groupby([groupingvar]).get_group(value))
        groupnames.append(value)
    for group in groups: #append all the values of **variable of interest**
        varlists.append(group[interestvar]) #to the list
    for var1 in varlists: #conduct the t-test, append all p-values of comparing the values
        for var2 in varlists:
            if (var1 is var2):
                continue
            else:
                results.append(stats.ttest_ind(var1,var2))
    for i in range(len(groupnames)): #display the p-values
        for j in range(len(groupnames)):
                if groupnames[i] is groupnames[j]:
                    continue
                else:
                    print('Compare '+ str(interestvar) + ' of ' + str(groupnames[i])+' and '+str(groupnames[j])+":\n")       
                    print(str(results[i]) + ":\n")
    return results[0] #For testing
        
hypothesis_test('Only child','Loneliness') #check t-test for
                                            #loneliness level of only child v.s. not only child
hypothesis_test('Gender','Loneliness') #check t-test for
                                            #loneliness level of male v.s. female




    
#%%

# -----  heat map plot distribution of loneliness -------
col = 'Loneliness'
corr = responses.corr() #find correlations
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#heatmap is too messy. thus, we are not using this graph


# -----  count plot distribution of loneliness -------  [figure_1]
var_of_interest = 'Loneliness'
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(responses[var_of_interest], orient = 'h')
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)


#%%
#------- query3: plot corr plot with seaborn --------  [figure_2a, figure_2]

def do_ploting(x, y, figsize): #draw only 1 bar
    fig, ax = plt.subplots(figsize= figsize)
    ax.set_title("Correlation coefficient of the variables")
    sns.barplot(x=x, y=y,  ax = ax)
    ax.set_ylabel("Correlation coefficients")

def correlation_plot(var_of_interest, df_main, figsize = (10,30)):
    def calc_corr(var_of_interest, df, cols, figsize): #calculate correlation
        lbls = []
        vals = []
        for col in cols:
            lbls.append(col)
            vals.append(np.corrcoef(df[col], df[var_of_interest])[0,1])
        corrs = pd.DataFrame({'features': lbls, 'corr_values': vals})
        corrs = corrs.sort_values(by='corr_values')
        do_ploting(corrs.corr_values, corrs['features'], figsize)
        return corrs
    #imputing the set
    df = copy.deepcopy(df_main)
    #df.replace(mapping, inplace = True)
    mean_values = df.mean(axis=0)
    df.fillna(mean_values, inplace=True)
    
    #correlating non-categorical varibales
    cols_floats = [col for col in df.columns if df[col].dtype!='object']
    cols_floats.remove(var_of_interest)
    corrs_one = calc_corr(var_of_interest, df, cols_floats, figsize)
    
    #correlating categorical variables
    cols_cats = [col for col in df.columns if df[col].dtype=='object']
    if cols_cats:
        df_dummies = pd.get_dummies(df[cols_cats])
        cols_cats = df_dummies.columns
        df_dummies[var_of_interest] = df[var_of_interest]
        corrs_two = calc_corr(var_of_interest, df_dummies, cols_cats, (5,10))
    else:
        corrs_two = 0
    return [corrs_one, corrs_two]

#plotting the correlation graph with var_of_interest(loneliness) & responses
corrs_area = correlation_plot(var_of_interest, responses)

#---------- The strongest correlations that we have are  -------------
corr_num = corrs_area[0] #for numeric variables
corr_cats = corrs_area[1] #for categorical variables

print(corr_num)
print(corr_cats)

# these are the max & min three features for corr_cats/ corr_num
#        corr_values     features
#85      0.245200                     Fake
#99      0.290439        Changing the past
#106     0.34468               Mood swings

#118    -0.467836        Happiness in life
#119    -0.358408            Energy levels
#103    -0.349509        Number of friends

#22     0.100282  Education_currently a primary school pupil
#16     0.111565              Internet usage_most of the day
#25     0.133602                    Education_primary school

#15     -0.135855     Internet usage_less than an hour a day
#18    -0.092580                                 Gender_male
#7     -0.088685             Punctuality_i am always on time


#%% query 4
# -------------- logreg to predict loneliness --------------------------
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#formatting string data for modeling


#create sub-dataset that will be used for predicting Loneliness
# not using all the variables.
# the variables we are using includes:
mov_mus   = df1.iloc[:,[0,19]]   #movie preferences
scared    = df1.iloc[:,63:73]    #phobias
interests = df1.iloc[:,31:63]    #hobbies & interests
demo      = df1.iloc[:,140:150]  #demographics
spending  = df1.iloc[:,133:140]  #spending habits

print(responses.columns.get_loc('Loneliness')) #find the column of loneliness
predict   = df1.iloc[:,99]       #predict var: loneliness

#join all the vars we are using:
df2 = mov_mus.join([scared, interests, demo, spending, predict])

#make the categorical variables to dummy variables:
gender  = pd.get_dummies(df2['Gender'])
handed  = pd.get_dummies(df2['Left - right handed'])
child   = pd.get_dummies(df2['Only child'])
vil_tow = pd.get_dummies(df2['Village - town'])
resid   = pd.get_dummies(df2['House - block of flats'])
educa   = pd.get_dummies(df2['Education'])

#drop the original dummy variables:
df2.drop(['Gender','Left - right handed','Only child','Village - town','House - block of flats','Education'], axis=1, inplace=True)
#join the dummy categorical variables:
df2 = df2.join([gender, handed, child, vil_tow, resid, educa])

#drop all nan
df2=df2.dropna()

#Instead of doing multi-label prediction, splitting Loneliness into two groups 
#- 3 or less, 4 or more
df2.loc[df2['Loneliness'] <= 3, 'Loneliness'] = 0
df2.loc[df2['Loneliness'] > 3, 'Loneliness'] = 1

#set up x & y data sets
x = df2.drop('Loneliness', axis=1)
y = df2['Loneliness']

#set up train, test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
#separate train test into 5 folds
kf = KFold(len(x_train), n_folds=5)

#using logistic regression to predict 
logreg = LogisticRegression()
param_grid = {'C':[.01,.03,.1,.3,1,3,10]}
gs_logreg = GridSearchCV(logreg, param_grid=param_grid, cv=kf)
gs_logreg.fit(x_train, y_train)
gs_logreg.best_params_ #{'C': 0.01}

#fit Logistic Regression model, eval scoring
logreg = LogisticRegression(C=.01)
logreg.fit(x_train, y_train)

print('Average accuracy score on cv (KFold) set: {:.3f}'.format(np.mean(cross_val_score(logreg, x_train, y_train, cv=kf))))
print('Accuracy score on test set is: {:.3f}'.format(logreg.score(x_test, y_test)))
#Average accuracy score on cv (KFold) set: 0.722
#Accuracy score on test set is: 0.701
#Not the best score, but want to keep single Logistic Regression model 
#so it's easy to evaluate features

# --------------- plot all features importance -------------  [figure_3a]
# put all coefficiets into a data frame & sort them.
coeff_df = pd.DataFrame(data=logreg.coef_[0], index=[x_train.columns], columns=['Feature_Import'])
coeff_df = coeff_df.sort_values(by='Feature_Import', ascending=False)

fig, ax1 = plt.subplots(1,1, figsize=(7,6)) #set up background
sns.barplot(x=coeff_df.index, y=coeff_df['Feature_Import'], ax=ax1)#set up background
ax1.set_title('All Features') #set up title
ax1.set_xticklabels(labels=coeff_df.index, size=6, rotation=90) #set up lable
ax1.set_ylabel('Importance') #set up lable

coeff_df['Feature_Import'].tail(10)         
#----------- plot the ten most positive & negative features. ---- [figure_3]
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,10))
sns.barplot(x=coeff_df.index[:10], y=coeff_df['Feature_Import'].head(10), ax=ax1)
ax1.set_title('Top Positive Features')
ax1.set_ylabel('Importance')
ax1.set_xticklabels(labels=coeff_df.index[:10], fontsize=8, rotation=20)


sns.barplot(x=coeff_df.index[-10:], y=coeff_df['Feature_Import'].tail(10), ax=ax2, palette='hls')
ax2.set_title('Top Negative Features')
ax2.set_ylabel('Importance')
ax2.set_xticklabels(labels=coeff_df.index[-10:], fontsize=8, rotation=20)


#%%
# ------------------- User Interation: Loneliness Test ------------------------------------
#the intercept of the logistic regression
logreg.intercept_ #0.00055438

# choose x variables :
# I chose variables that: abs(coefficient) > 0.09 
# There are 4 positive variables & 4 negative variables.

# equation for predicting loneliness: 
#yscore=0.00055438+0.133*Fear of Public Speaking + 0.108*writing 
        # + 0.0924* PC + 0.0919* Internet - 0.102* Fun with friends 
        # - 0.102* Economy Management - 0.11* Cars - 0.128* Entertainment spending
        
def areyoulonely():
    print(' ====== LONELINESS TEST  =======') #tab enter to begin
    print('Do you think you are lonely?')
    temp=input() #ignore any input
    print('')
    print('Doesn\'t matter what you think.')
    print('We will see after 8 questions.')
    
    #store user input into each variable:
    x1_public_speaking = int(input('1. Public speaking: Not afraid at all 1-2-3-4-5 Very afraid of (integer)'))
    x2_writing=int(input('2. Poetry writing: Not interested 1-2-3-4-5 Very interested (integer)'))
    x3_internet=int(input('3. Internet: Not interested 1-2-3-4-5 Very interested (integer)'))
    x4_PC=int(input('4. PC Software, Hardware: Not interested 1-2-3-4-5 Very interested (integer)'))
    
    x5_fun_fri=int(input('5. Socializing: Not interested 1-2-3-4-5 Very interested (integer)'))
    x6_eco_man=int(input('6. Economy, Management: Not interested 1-2-3-4-5 Very interested (integer)'))
    x7_cars=int(input('7. Cars: Not interested 1-2-3-4-5 Very interested (integer)'))
    x8_ent_sp=int(input('8. I spend a lot of money on partying and socializing.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)'))    
    #call calulate loneliness function:
    calculateloneliness(x1_public_speaking,x2_writing,x3_internet,x4_PC,x5_fun_fri,x6_eco_man,x7_cars,x8_ent_sp)
    return x1_public_speaking,x2_writing,x3_internet,x4_PC,x5_fun_fri,x6_eco_man,x7_cars,x8_ent_sp

def calculateloneliness(x1,x2,x3,x4,x5,x6,x7,x8):
    #set up the equation:
    yscore=0.00055438+0.133*x1+0.108*x2+0.0924*x3+0.0919*x4-0.102*x5-0.102*x6-0.11*x7-0.128*x8
    print()
    print(' ==========  RESULT  =========')
    if yscore>0.5: #if probability>0.5, then user is lonely
        print('Hi Mr./Ms. Lonely !')
        #print the probability in percent with 2 decimal places
        print('You are '+str('{percent:.2%}'.format(percent=yscore)+' likely to be a lonely person.') )
    else:# else, not lonely
        print('You must be a joyful person!')
        #print the probability in percent with 2 decimal places
        print('You are only'+str('{percent:.2%}'.format(percent=yscore)+' likely to be a lonely person.'))
    outputfortest=int(yscore)
    return outputfortest
#call the areyoulonely() function to start test
areyoulonely()
 
        


