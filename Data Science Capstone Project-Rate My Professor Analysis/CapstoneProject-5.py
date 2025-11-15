#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:10:47 2024

@author: zhangjiahao
"""
#initiate random generator with seed number
import numpy as np
import random

N_number = 14758736
np.random.seed(N_number)
random.seed(N_number)



import pandas as pd

num_file_path = '/Users/zhangjiahao/PrincipleDS/Project/rmpCapstoneNum.csv'
num_data = pd.read_csv(num_file_path, header=None)

qual_file_path = '/Users/zhangjiahao/PrincipleDS/Project/rmpCapstoneQual.csv'
qual_data = pd.read_csv(qual_file_path, header = None)


#add column name for the file with number (ratings)
num_data.columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a Pepper", "Proportion Take Again", "Number of Online Ratings",
    "Male Gender", "Female Gender"
]

#add column name for the file with University info
qual_data.columns = ["Major/Field", "University", "US State"]

#check file and general descriptive statistics
print(num_data.head())
print(qual_data.head())   
print(num_data.describe())
print(qual_data.describe())

# merge two data sheet into one, easier for manupulation
merged_data = pd.concat([num_data, qual_data], axis=1)
print(merged_data.describe(include="all"))



#Data Clearning, first see how many cells for each column are null (see if the entire row are Nan)
print(merged_data.isnull().sum())



#noticed that the following columns all have nan value on the same row
check1=merged_data[merged_data[["Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a Pepper", "Number of Online Ratings", "Major/Field", "University", "US State"] ].isnull().all(axis=1)]
                        
                        
print(check1)                     
                        
                        

#Then we drop those rows since there are just nan value and meaningless
merged_data_cleaned = merged_data.dropna(subset=["Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a Pepper", "Number of Online Ratings", "Major/Field", "University", "US State"])

merged_data_cleaned.isnull().sum()


# Handle missing values in "Proportion Take Again" by filling them with the median
# This avoids dropping rows unnecessarily as this column might be useful later
merged_data_cleaned["Proportion Take Again"] = merged_data_cleaned["Proportion Take Again"].fillna(merged_data_cleaned["Proportion Take Again"].median())


#check the clearned data again.
print(merged_data_cleaned.describe())


###### !!!!!!!!!!!
# important step! Ihe project prompt, consideration part, professor asked to handle 
# the fact that "it is more meaningful if the average rating is based on more ratings
# For my consideration, accepting all data will result extreme rating by just one rating,
# and this might caused by students' bias. And for the method "Weight the average rating,
# Using the number of ratings as weights introduces the information about the quantity of ratings 
# into the "rating" variable. This can lead to inaccurate analysis because the subsequent evaluation of 
# "rating" may be confounded by the additional information from the number of ratings.


# Below was my attempt to test what the data will be like using the metho "Weight the average"
# I set the weight for each average rating as its number of rating/total number of rating
# but this introduce confounding effect as subsequent evaluation of rating" may be confounded by the 
# additional information from the number of ratings

Weighted_merged_data_cleaned = merged_data_cleaned.copy()

Weighted_merged_data_cleaned["Weighted Average Rating"] = Weighted_merged_data_cleaned["Average Rating"] * Weighted_merged_data_cleaned["Number of Ratings"] / merged_data_cleaned["Number of Ratings"].sum()

print(Weighted_merged_data_cleaned["Weighted Average Rating"])

### Conclusion, thus I used the set a threshold, and set the threshold as 5 number of ratings.
merged_data_cleaned = merged_data_cleaned[merged_data_cleaned['Number of Ratings'] >= 5]



# Question 1
print()
print("Question 1")
print()

from scipy.stats import shapiro

# Split data into male and female groups
male_ratings = merged_data_cleaned[merged_data_cleaned["Male Gender"] == 1]["Average Rating"]
female_ratings = merged_data_cleaned[merged_data_cleaned["Female Gender"] == 1]["Average Rating"]

print("median of male Average rating:",male_ratings.median())
print("median of female Average rating:",female_ratings.median())

# Test normality of the distributions for male and female ratings
shapiro_test_male = shapiro(male_ratings)
shapiro_test_female = shapiro(female_ratings)

#or test whether both data is normally distributed by graph

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

sns.kdeplot(male_ratings, label="Male Ratings", shade=True, color="blue")
sns.kdeplot(female_ratings, label="Female Ratings", shade=True, color="red")

plt.title("Distribution of Avarage Raing")

plt.legend()
plt.show()


# Print p-values of normality tests
#both group are not normally distributed, thus using nonparametric test
print(shapiro_test_male.pvalue, shapiro_test_female.pvalue)

#I planned to use mannwhitneyu test as:
from scipy.stats import mannwhitneyu
u_stats, u_p_value = mannwhitneyu(male_ratings, female_ratings)
"Mann-Whitney U-test", u_stats, u_p_value

print("u_stats ",u_stats)
print("u_p_value ",u_p_value)

#conclusion: Since the p value is extremely small, we drop the null hypothesis and can say that there is statistical evidence that
#the male professor's average rating is higher than female.
    

# Not done yet, since other factors like "Average Difficulty", "Number of Ratings", "Proportion Take Again"
# might also incluces the average number of rating, we need to examine strength of other factors.


import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt



X = merged_data_cleaned[["Male Gender", "Female Gender", "Average Difficulty", "Number of Ratings","Received a Pepper","Number of Online Ratings", "Proportion Take Again"]]
y = merged_data_cleaned["Average Rating"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("Below is the model summary of the un-normalized model (I will normalize the factor and make them into same scale")
print(model.summary())



# Create a copy of the cleaned dataset for normalization
normalized_data = merged_data_cleaned.copy()
columns_to_normalize = ["Male Gender","Female Gender", "Average Difficulty", "Number of Ratings","Received a Pepper","Number of Online Ratings", "Proportion Take Again"]

scaler = MinMaxScaler()
# Apply normalization to the selected columns
normalized_data[columns_to_normalize] = scaler.fit_transform(normalized_data[columns_to_normalize])
print("normalized data:", normalized_data[columns_to_normalize].head())

# nomalize data to ensure the factors are in the same scale, so we can compare"
X = normalized_data[columns_to_normalize]
y = normalized_data["Average Rating"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("below is the summary of this normalized multiple regression model (Not dealing with Colinearity yet, next model will be), Q1")
print(model.summary())

from sklearn.linear_model import Ridge


ridge_model = Ridge(alpha=100)
ridge_model.fit(X, y)

y_pred = ridge_model.predict(X)
r_squared = ridge_model.score(X, y)
# rmse = mean_squared_error(y, y_pred, squared=False)

ridge_coefficients = ridge_model.coef_
ridge_intercept = ridge_model.intercept_

print(f"Regression Equation After Lasso: y = {ridge_intercept:.4f} + " + " + ".join([f"{slope} * {col}" for slope, col in zip(ridge_coefficients, X.columns)]))






# here, we make two scatter plott to visualize the effect of male gender on average rating,
# and we find out that though there is apositive relationshi, but relationship is quite weak (as coeeficient is....)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Male Gender', y="Average Rating", data=merged_data_cleaned)
sns.lineplot(x=merged_data_cleaned["Male Gender"], y=model.fittedvalues, color="red")

plt.show()

# In fact, the Average Difficulty has a stronger negative relationship with Average rating,
# as the Average Difficulty increases, the average rating goes down. 
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Average Difficulty', y="Average Rating", data=merged_data_cleaned)
sns.lineplot(x=merged_data_cleaned["Average Difficulty"], y=model.fittedvalues, color="red")

plt.show()


#As for the impact of gender: the score of male professors is indeed significantly higher than that of female professors
#but the absolute value of the regression coefficient is very small; 
#Grades are driven more by the difficulty of the course, the number of grades, and the proportion of students who choose the course again.

#########Question 2
print()
print("Question 2")
print()

num_median = merged_data_cleaned["Number of Ratings"].median()
high_num_ratings = merged_data_cleaned[merged_data_cleaned["Number of Ratings"] > num_median]["Average Rating"]
low_num_ratings = merged_data_cleaned[merged_data_cleaned["Number of Ratings"] <= num_median]["Average Rating"]

# Test Normallity of distribution
shapiro_test_high = shapiro(high_num_ratings)
shapiro_test_low = shapiro(low_num_ratings)

print(shapiro_test_high.pvalue, shapiro_test_low.pvalue)

# Visualize the distribution, and we can see both group are not normally distributed
plt.figure(figsize=(12, 6))

sns.kdeplot(high_num_ratings, label="High Num Ratings", shade=True, color="blue")
sns.kdeplot(low_num_ratings, label="Low Num Ratings", shade=True, color="red")

plt.title("Distribution of Avarage Rating")

plt.legend()
plt.show()

#find out that there is a true difference of two group since p-value...., thus the experience indeed
#makes the avergae rating higher, but we still need to test the strength of this effect, so we do a regression model.
from scipy.stats import mannwhitneyu
u_stats, u_p_value = mannwhitneyu(high_num_ratings, low_num_ratings)
print()
print("Question 2 Mann-Whitney U-test", "Q2 U stats",u_stats, "Q2 U_p_value", u_p_value)
print()


# Here we use this regression model to see the strength of exprience or not on the Average rating,
# and we find out that though it poses a positive effect as the exprience increases and the average rating (quality of teaching) increases too
# the effect is very small.
import statsmodels.api as sm

X = merged_data_cleaned['Number of Ratings']
y = merged_data_cleaned["Average Rating"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

#Shows from the graph, though the regression line looks steep, there is a postive relationship between Number of ratings and Avergae difficulty
#The Number of Ratings increases about 100, the Average rating increase 0.5.
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Number of Ratings', y="Average Rating", data=merged_data_cleaned)
sns.lineplot(x=merged_data_cleaned['Number of Ratings'], y=model.fittedvalues, color="red")

plt.show()


#Question 3
print()
print("Question 3")
print()

#First, I used a Linregress model to see how much variance in the avergae rating was explained by average difficulty
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(merged_data_cleaned["Average Rating"], merged_data_cleaned["Average Difficulty"])

print({
    "Slope": slope,
    "Intercept": intercept,
    "R_value": r_value,
    "R-squared": r_value**2,
    "P-value": p_value,
    "Standard Error": std_err
})




#这块图不知道画的对不对
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Average Difficulty', y="Average Rating", data=merged_data_cleaned)
sns.lineplot(x=merged_data_cleaned["Average Difficulty"], y=model.fittedvalues, color="red")

plt.show()


#R-squared is only 0.3832, which is small, then I planned to use a spearman as it not requiring the relationship to be linear.
from scipy.stats import spearmanr




spearman_corr, spearman_p_value = spearmanr(merged_data_cleaned["Average Rating"], merged_data_cleaned["Average Difficulty"])

print({
    "Spearman Correlation": spearman_corr,
    "P-value": spearman_p_value
})

#Conclusion
#We say the Spearman correlation coefficient (-0.6027) indicates a moderate negative correlation
#of moderate strength, and the p_value (0.0) indicates a statistically significant correlation between the two variables.
#But the correlation of the Negative linear relationship is still a little bit stronger


#Question 4:
print()
print("Question 4")
print()
#To determine whether there is a difference in average rating between the group of professpr 
#who teach a low online modality or not teaching a lot online modality, I first split the
#data by median into two group, and this ensures same number of professor in two sample group too.



from scipy import stats
# Find median value for "Number of Online Ratings"
meadian_online_ratings = merged_data_cleaned["Number of Online Ratings"].median()


#Notice, not equal size. the median is 0.
print(meadian_online_ratings)


# then I decide to test the percetage of professor who received 1,2,4 or more ratings and setting threshold for groups
low_threshold = merged_data_cleaned["Number of Online Ratings"].quantile(0.2)  # Lower 20% threshold
# still low threshold is 0 too, so we set low experience group as number of rating = 0.
print(low_threshold)
high_threshold = 2

total_professors = len(merged_data_cleaned)
professors_with_online_ratings = len(merged_data_cleaned[merged_data_cleaned["Number of Online Ratings"] >= high_threshold])
percentage_with_online_ratings = (professors_with_online_ratings / total_professors) * 100
print(f"Percentage of professors with online ratings: {percentage_with_online_ratings:.2f}%")




# Split into two groups: low online modality and high online modality
low_online_group = merged_data_cleaned[merged_data_cleaned["Number of Online Ratings"] <=low_threshold]
high_online_group = merged_data_cleaned[merged_data_cleaned["Number of Online Ratings"] >= high_threshold]



# Test for normality in both groups
shapiro_test_low = stats.shapiro(low_online_group["Average Rating"])
shapiro_test_high = stats.shapiro(high_online_group["Average Rating"])
print(shapiro_test_low.pvalue, shapiro_test_high.pvalue)
#They are not normally distributed, thus we need to use an nonparametric test 
#Since we are not interested in how two group are distributed, we use a Mann-Whitney U-test to compare median.


#make a distribution graph of density of Average rating and Online Modality
plt.figure(figsize=(10, 6))

sns.kdeplot(low_online_group["Average Rating"], label="Low Online Modality", shade=True, color="blue")
sns.kdeplot(high_online_group["Average Rating"], label="High Online Modality", shade=True, color="red")

plt.title("Distribution of Average Ratings by Online Teaching Modality")
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.legend()
plt.show()

#test the proportion of professor who received the online rating base on the fact that the median for number of rating is 0
#Finding the threshold for "Lot of online modality"




print(low_online_group["Number of Online Ratings"].median())
print(high_online_group["Number of Online Ratings"].median())


#Median Average rating of low online group"
print("Median Average rating of low online group",low_online_group["Average Rating"].median())
#Median Average rating of High online group"
print("Median Average rating of High online group",high_online_group["Average Rating"].median())

u_stats, u_p_value = stats.mannwhitneyu(low_online_group["Average Rating"], high_online_group["Average Rating"])
print("Mann-Whiteney U-test", "u_stats:",u_stats,"u_p_value:", u_p_value)

#We can see that the p_value is extremelly small, thus we can drop the null hypothesis and say there is a statistical evidence 
#who teach a lower proportion of their classes online receive higher ratings compared to those who teach fewer or no online classes. 

'''

## Question 5
print()
print("Question 5")
print()
merged_data_5 = merged_data.dropna(subset=["Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a Pepper", "Number of Online Ratings", "Major/Field", "University", "US State"])
merged_data_5 = merged_data_5.dropna(subset=["Proportion Take Again"])

X = merged_data_5["Proportion Take Again"]
y = merged_data_5["Average Rating"]

slope, intercept, r_value, p_value, std_err = linregress(X, y)

print("slope:",slope, "intercept:",intercept, "r_value:",r_value, "r_square:",r_value**2, "p_value:",p_value, "std_err:",std_err)

from scipy.stats import pearsonr

correlation, p_value = pearsonr(merged_data_5["Average Rating"], merged_data_5["Proportion Take Again"])
print("correlation:",correlation, "p_value:",p_value)
            
plt.figure(figsize=(12, 6))

# Scatter plot of data points
sns.scatterplot(x="Proportion Take Again", y="Average Rating", data=merged_data_5, alpha=0.6)

# Regression line
sns.lineplot(x=merged_data_5["Proportion Take Again"], 
             y=(slope * merged_data_5["Proportion Take Again"] + intercept), 
             color="red", label="Regression Line")

# Add labels and title
plt.title("Relationship Between Average Rating and Proportion Take Again")
plt.xlabel("Proportion Take Again")
plt.ylabel("Average Rating")
plt.legend()

plt.show()




#Question 6
print()
print("Question 6")
print()
#Split the data to Professor who are "Hot" (Received a Pepper) and Professor who are not "Hot" (no Pepper) 
hot_group = merged_data_cleaned[merged_data_cleaned["Received a Pepper"] == 1]["Average Rating"]
not_hot_group = merged_data_cleaned[merged_data_cleaned["Received a Pepper"] == 0]["Average Rating"]

#Test the distribution
shapiro_hot = stats.shapiro(hot_group)
shapiro_not_hot = stats.shapiro(not_hot_group)

print(shapiro_hot.pvalue, shapiro_not_hot.pvalue)


plt.figure(figsize=(12, 6))

#  plot for "hot" professors
sns.kdeplot(hot_group, label="Hot Professors", shade=True, color="orange")

#  plot for "not hot" professors
sns.kdeplot(not_hot_group, label="Not Hot Professors", shade=True, color="blue")

plt.title("Distribution of Average Ratings for Hot vs. Not Hot Professors")
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.legend()

plt.show()

u_stat, u_p_value = stats.mannwhitneyu(hot_group, not_hot_group)
print("Mann-Whiteney U-test", "Q6_u_stat:",u_stat, "Q6_u_p_value:", u_p_value)
print("Median of the hot group:",hot_group.median())
print("Median of the Not hot group:",not_hot_group.median())




#question 7 
print()
print("Question 7")
print()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = merged_data_5[["Average Difficulty"]].values
y = merged_data_5["Average Rating"].values

#Split the data to ensure not over fitt.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the regression model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r_squared_train = model.score(X_train, y_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)

r_squared_test = model.score(X_test, y_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)


intercept = model.intercept_
slope = model.coef_[0]

print(f"Regression Equation: y = {intercept:.4f} + {slope:.4f} * x")
print(f"R-squared: training : {r_squared_train}, test : {r_squared_test}")
print(f"RMSE: train : {rmse_train}, test : {rmse_test}")

plt.figure(figsize=(12, 6))

# Scatter plot of actual test data
sns.scatterplot(x=X_test.flatten(), y=y_test, alpha=0.6, label="Test Data Points")

# Regression line
sns.lineplot(x=X_test.flatten(), y=y_pred_test, color="red", label="Regression Line (Test)")

# Add title and labels
plt.title("Linear Regression: Average Rating vs. Average Difficulty (Test Set)")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.legend()

plt.show()

## the r-square of single factor is too low,the explanation power is low (Explain R-square)



## Question 8, go back and check colinearity
print()
print("Question 8")
print()
y = merged_data_5["Average Rating"]
X = merged_data_5[["Average Difficulty", "Number of Ratings",
    "Received a Pepper", "Proportion Take Again", "Number of Online Ratings",
    "Male Gender", "Female Gender"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r_squared_train = model.score(X_train, y_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)

r_squared_test = model.score(X_test, y_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

intercept = model.intercept_
slopes = model.coef_

print(f"Regression Equation: y = {intercept:.4f} + " + " + ".join([f"{slope} * {col}" for slope, col in zip(slopes, X.columns)]))
print(f"R-squared: training : {r_squared_train}, test : {r_squared_test}")
print(f"RMSE: train : {rmse_train}, test : {rmse_test}")

#Plot the coeffcieint bar graph before dealing with colinearity
variables = ["Average Difficulty", "Number of Ratings", "Received a Pepper", 
             "Proportion Take Again", "Number of Online Ratings", "Male Gender", "Female Gender"]
standard_coefficients = slopes

# Plot comparison of coefficients
plt.figure(figsize=(14, 6))

# Standard Regression Coefficients
plt.subplot(1, 2, 1)
plt.bar(variables, slopes, alpha=0.7, color='blue')
plt.title("Coefficients: Standard Regression")
plt.xlabel("Independent Variables")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)

# Colinearity:
import pandas as pd

correlation_matrix = pd.DataFrame(X).corr()



#Since we consider all predictors are potentially important, I do not want to exclude any variable from model, thus consider use Ridge.
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=100)
ridge_model.fit(X, y)

y_pred = ridge_model.predict(X)
r_squared = ridge_model.score(X, y)
rmse = mean_squared_error(y, y_pred, squared=False)

ridge_coefficients = ridge_model.coef_
ridge_intercept = ridge_model.intercept_

print(f"Ridge Regression Equation: y = {ridge_intercept:.4f} + " + " + ".join([f"{slope} * {col}" for slope, col in zip(ridge_coefficients, X.columns)]))
print(f"R-squared: training : {r_squared_train}, test : {r_squared_test}")
print(f"RMSE: train : {rmse_train}, test : {rmse_test}")

# Ridge Regression Coefficients
plt.subplot(1, 2, 2)
plt.bar(variables, ridge_coefficients, alpha=0.7, color='green')
plt.title("Coefficients: Ridge Regression")
plt.xlabel("Independent Variables")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)

# Adjust layout for better visualization
plt.tight_layout()
plt.show()




# percentage of change from standard coefficients to ridge coeffcients

# Coefficients from Standard Regression
standard_coefficients = slopes

# Coefficients from Ridge Regression
ridge_coefficients = ridge_model.coef_

variables = ["Average Difficulty", "Number of Ratings", "Received a Pepper", 
             "Proportion Take Again", "Number of Online Ratings", "Male Gender", "Female Gender"]

# Calculate Percentage Change
percentage_change = ((ridge_coefficients - standard_coefficients) / standard_coefficients) * 100

# Create DataFrame for visualization
coefficients_comparison = pd.DataFrame({
    "Variable": variables,
    "Standard Coefficients": standard_coefficients,
    "Ridge Coefficients": ridge_coefficients,
    "Percentage Change (%)": percentage_change
})

# Plot Percentage Change
plt.figure(figsize=(14, 6))
plt.bar(coefficients_comparison["Variable"], coefficients_comparison["Percentage Change (%)"], color='purple', alpha=0.7)
plt.title("Percentage Change in Coefficients After Ridge Regression")
plt.xlabel("Variables")
plt.ylabel("Percentage Change (%)")
plt.xticks(rotation=45)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()

# Plot Absolute Change
absolute_change = ridge_coefficients - standard_coefficients
plt.figure(figsize=(14, 6))
plt.bar(coefficients_comparison["Variable"], absolute_change, color='orange', alpha=0.7)
plt.title("Absolute Change in Coefficients After Ridge Regression")
plt.xlabel("Variables")
plt.ylabel("Absolute Change")
plt.xticks(rotation=45)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()







#Question 9
print()
print()
print("Question 9")
print()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

X = merged_data_5[['Average Rating']]
y = merged_data_5["Received a Pepper"]
#Checked distribution to see is it imbalanced (result, no pepper: Pepper = 6:4)
class_distribution = y.value_counts(normalize=True)

print(class_distribution)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Use SMOTE to resample equal sized sample (balancing)
smote = SMOTE()
X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
print(y_train_resample.value_counts(normalize=True))


# use logistic regression, the simplest one, see performance (ROC AUC is high)
logistic_model = LogisticRegression()
# logistic_model = LogisticRegression(class_weight="balanced")
logistic_model.fit(X_train_resample, y_train_resample)

y_pred_train = logistic_model.predict(X_train_resample)
y_pred = logistic_model.predict(X_test)
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]
y_pred_prob_train = logistic_model.predict_proba(X_train_resample)[:, 1]


roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc_train = roc_auc_score(y_train_resample, y_pred_prob_train)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_metrics = classification_report(y_test, y_pred, output_dict=True)




# High ROC AUC, model have good performance, train and test ROC AUC is similar, no overfitting
print({
    "Class Distribution" : class_distribution.to_dict(),
    "Train ROC AUC" : roc_auc_train,
    "ROC AUC" : roc_auc,
    "Confusion Matrix" : conf_matrix.tolist(),
    "Classification Metrics" : classification_report
})

print("Confusion Metrix Q9:")
print(conf_matrix)
print("Classification_metrics Q9")
print(classification_report(y_test, y_pred))


# plot the ROC Curve

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="red", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, alpha=0.2, color="red")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle='--', label="random")
plt.legend()
plt.grid()
plt.show()


#Q10
print()
print()
print("Question 10")
print()

X = merged_data_5[["Average Rating", "Male Gender", "Female Gender", "Average Difficulty", "Number of Ratings", "Number of Online Ratings", "Proportion Take Again"]]
y = merged_data_5["Received a Pepper"]

class_distribution = y.value_counts(normalize=True)

#check distribution to see is it imbalanced, result no pepper : pepper = 6:4
print(class_distribution)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Use SMOTE to resample equal sized sample (balancing)
smote = SMOTE()
X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)

# Check class distribution after resampling
print("Class Distribution After Resampling:", y_train_resample.value_counts(normalize=True))


# similarly, use logistic model
logistic_model = LogisticRegression(penalty='l2', C=1.0) # or apply dealing with colinearity
#logistic_model = LogisticRegression()
# or I can use logistic_model = LogisticRegression(class_weight="balanced")
logistic_model.fit(X_train_resample, y_train_resample)


y_pred = logistic_model.predict(X_test)
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_metrics = classification_report(y_test, y_pred, output_dict=True)



print({
    "Class Distribution" : class_distribution.to_dict(),
    "ROC AUC" : roc_auc,
    "Confusion Matrix" : conf_matrix.tolist(),
    "Classification Metrics" : classification_report
})

print("Confusion Metrix Q10:")
print(conf_matrix)
print("Classification_metrics Q10:")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="red", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, alpha=0.2, color="red")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle='--', label="random")
plt.legend()
plt.grid()
plt.show()

#Extra Credit
#logistic: Group by State, and see is there a difference in the mean of Average rating an rank by state.

statewise_anlaysis = merged_data_cleaned.groupby("US State").agg({
    "Average Rating" : "mean"
})

statewise_anlaysis_sorted = statewise_anlaysis.sort_values(by="Average Rating", ascending=False).head(10)
print(statewise_anlaysis_sorted)

#Noticed the mean value seems different of each state.
import matplotlib.pyplot as plt

statewise_anlaysis_sorted.reset_index(inplace=True)

plt.figure(figsize=(10, 6))
plt.barh(statewise_anlaysis_sorted["US State"], statewise_anlaysis_sorted["Average Rating"], color="skyblue")
plt.gca().invert_yaxis()
plt.show()



from scipy.stats import kruskal

print(kruskal(*[group["Average Rating"].values for name, group in merged_data_cleaned.groupby("US State")]))
'''
