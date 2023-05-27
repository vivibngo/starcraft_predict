# %% [markdown]
# Starcraft Rank Prediction by
# Vivi B. Ngo

# %%
#Libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy.stats import shapiro
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %% [markdown]
# # ETL

# %%
starcraft_df = pd.read_csv('/Users/vivib.ngo/desktop/starcraft_player_data.csv')

# %%
starcraft_df

# %%
#Remove irrelavent columns
starcraft_df = starcraft_df.drop('GameID', axis=1)

# %%
#Count how many non-int rows there are
non_integer_rows = starcraft_df[~starcraft_df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all(), axis=1)]
non_integer_rows
len(non_integer_rows)

# %% [markdown]
# The Professional Leagues coded 1-8 are not actual ranks within Starcraft, the highest rank for Starcraft is Grandmaster, hence it will be removed, however if was needed to be included then imputation method could be used to replace the missing values. 

# %%
#Remove rows contain '?'
newstar_df = starcraft_df[~starcraft_df.isin(['?']).any(axis=1)]
newstar_df

# %% [markdown]
# # EDA

# %%
for i in range(1, 9):
    length = len(starcraft_df[starcraft_df['LeagueIndex'] == i])
    print(f"Length of LeagueIndex {i}: {length}")

# %%
for i in range(1, 9):
    length = len(newstar_df[newstar_df['LeagueIndex'] == i])
    print(f"Length of LeagueIndex {i}: {length}")

# %%

# Distribution of Ranks
plt.figure(figsize=(8, 6))
sns.countplot(data=newstar_df, x='LeagueIndex')
plt.title('Distribution of Ranks')
plt.xlabel('Rankings')
plt.ylabel('Count')
plt.show()

# %%
# Perform Shapiro-Wilk test
statistic, p_value = shapiro(newstar_df['LeagueIndex'])

# Print the test statistic and p-value
print("Shapiro-Wilk Test:")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05  # significance level
if p_value > alpha:
    print("The ranking variable follows a normal distribution.")
else:
    print("The ranking variable does not follow a normal distribution.")

# %%
# Rank vs. Other Variables
variables_of_interest = ['ActionLatency', 'APM', 'NumberOfPACs']

for variable in variables_of_interest:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=newstar_df, x='LeagueIndex', y=variable)
    plt.title(f'{variable} by Rank')
    plt.xlabel('Rankings')
    plt.ylabel(variable)
    plt.show()

# %% [markdown]
# The higher the rank, the less outliers there are. The higher rank are more consistent in their performance, they're consistent in their performance and more consistent in metrics that would be considered good behavior or actions. The lower the variation, the more consistent the higher players are performing.

# %%
print(newstar_df['HoursPerWeek'].dtype)
print(newstar_df['TotalHours'].dtype)
newstar_df['HoursPerWeek'] = newstar_df['HoursPerWeek'].astype(int)
newstar_df['TotalHours'] = newstar_df['TotalHours'].astype(int)

# %%
# Plotting the data
plt.figure(figsize=(10, 6))
sns.lineplot(data=newstar_df, x='LeagueIndex', y='HoursPerWeek', color='blue', label='HoursPerWeek')
sns.lineplot(data=newstar_df, x='LeagueIndex', y='TotalHours', color='red', label='TotalHours')
plt.title('Player Performance over Time')
plt.xlabel('LeagueIndex')
plt.ylabel('Hours')
plt.legend()
plt.show()


# %% [markdown]
# # Analysis
# Higher ranked players tend to have more hours dedicated to the game, which makes sense. At higher ranks such as grandmaster, players do not need to dedicated so many total hours overall to increase their rank if their main concern is to avoid elo decay, which we see the case for master and grandmaster

# %%
# Rank Patterns over Time (if applicable)
# Custom mapping of rankings to labels
# Sort the DataFrame by 'LeagueIndex'
newstar_df_sorted = newstar_df.sort_values('LeagueIndex')

plt.figure(figsize=(10, 10))
sns.lineplot(data=newstar_df_sorted, x='LeagueIndex', y='HoursPerWeek')
plt.title('Rank Patterns over Time')
plt.xlabel('LeagueIndex')
plt.ylabel('HoursPerWeek')

plt.show()

# %%
# Correlation Analysis : Pick variables of interest
correlation_matrix = newstar_df[['LeagueIndex', 'APM', 'ActionLatency', 'NumberOfPACs']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# # Analysis

# %% [markdown]
# APM, NumberOfPACs, and LeagueIndex have a positive correlation coefficient, which indicates a positive linear relationship, meaning that as APM increases, LeagueIndex increases proportionally. ActionLatency and LeagueIndex have a negative linear relationship, meaning that as ActionLatency decreases, LeagueIndex increases. 

# %%
# Rank Comparison
rank_comparison = newstar_df.groupby('LeagueIndex').mean()
rank_comparison

# %% [markdown]
# # Analysis
# The higher the average APM, NumberOfPACs, and lower ActionLatency, shows a an increase in rank/LeagueIndex. This is expected, players who are more skilled should have a lower reaction time and higher action per minute, being able to efficiently think about a counterplay/contingecny plan and executing are good indicators of a well seasoned player. NumberOfPACs could be seen as map awareness and how often the player is looking around, the panning of the map allows the player to gain more map awareness, when the player has more gains more information due to the awareness of the map, the higher their rank tends to be. 

# %% [markdown]
# # Feature Selection

# %% [markdown]
# The code uses a random forest classifier to train a supervised learning model on the dataset. It retrieves the feature importances from the trained model. These importances represent the relative importance of each feature in making predictions.  

# %%
# Separate the predictor variables (X) and the target variable (y)
X = newstar_df.drop('LeagueIndex', axis=1)  # Adjust the column name if needed
y = newstar_df['LeagueIndex']

# %%
X

# %%
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
# Fit the classifier on the data
rf_classifier.fit(X, y)
# Get feature importances
importances = rf_classifier.feature_importances_
# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance values in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# %% [markdown]
# A random forest classifier was used because we're trying to predict the rank/league index based on a set of variables. Random Forest is commonly used for classification tasks where the goal is to predict the class/category of an observation based on a set of input variables

# %%
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Variable Importance')
plt.show()

# Print the feature importance rankings
print(feature_importance_df)

# %% [markdown]
# # Random Forest Model
# 
# A random forest classifier was chosen for predicting player rankings based on performance since it's good at handling complex relationships, player performance data often contains intricate relationships and interactions between various perofrmance metrics. Random forest can effectively capture and model these complex relationships.

# %%
# Get the first tree from the Random Forest classifier
selected_features = ['APM', 'ActionLatency', 'NumberOfPACs', 'GapBetweenPACs', 'TotalHours', 'SelectByHotkeys', 'AssignToHotkeys', 'WorkersMade', 'ActionsInPAC', 'MinimapAttacks', 'MinimapRightClicks', 'TotalMapExplored']
X2 = X[selected_features]

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

# %%
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)

# %% [markdown]
# # Analysis
# Precision is the proportion of correctly predicted instances of a specific LeagueIndex. A precision of 0.44 for LeagueIndex of 1 (Bronze) means that 44% of the instances predicted as LeagueIndex 1 were actually LeagueIndex 1(Bronze).
# 
# Recall is known as true positive rate, its the proportion of correctly predicted instances of a specific rank our of all the instances that actually belong to that rank. For example a recall of 0.33 for LeagueIndex 1(Bronze) means that 33% of the actual instances of LeagueIndex 1 were correctly predicted as Bronze.
# 
# The F1-score is the harmonic mean of precision and recall, it is a single metric that combines precision and recall. It is often used when there is an imbalance between class. 
# 
# Support refers to the number of instances in each rank in the test data.
# 
# Accuracy is the overall proportion of correctly predicted instances across all classes. It measures the overall performance of the model. The model is 43% accurate.
# 
# Macro average alculates the average performance across all ranks, giving equal weight to each rank, it provides an overall evaluation of the model's performance without considering rank imbalance.
# 
# Weighted average calculates the average performance across all ranks, but takes into account the number of instances of each rank. It provides an evaluation that considers rank imbalance by giving more weight to ranks with more instances.

# %% [markdown]
# # Tree Plot

# %%
from sklearn import tree

# Get the first tree from the Random Forest classifier
selected_features = ['APM', 'ActionLatency', 'NumberOfPACs']
X_selected = X[selected_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(max_depth = 3)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)

# Get one of the trees from the Random Forest classifier (e.g., the first tree)
tree_estimator = rf_classifier.estimators_[0]

# configure plot size and spacing
plt.figure(figsize=(20, 20), dpi=1000)
# plt.subplots_adjust(left=0.1, right=1, top=1, bottom=0.8)
#plot the tree
tree.plot_tree(tree_estimator, feature_names=selected_features, class_names=['1', '2', '3', '4', '5', '6', '7'], filled=True, max_depth=3, fontsize=10)

plt.show()

# %% [markdown]
# # Analysis

# %% [markdown]
# If the ActionLatency is <= 56.705, the tree will follow the left branch from this node. The Gini impurity score = 0.807 suggests that the samples in this node are distributed across multiple LeagueIndex rather than being predominantly in a single LeagueIndex. Samples = 1674 represents the number of samples(data points) that reached this node during the training processes. The value = [118, 277, 482, 634, 639, 490, 30] shows the distribution of target LeagueIndex, the values correspond to the counts of each LeagueIndex Rank, for example there are 118 samples with LeagueIndex 1.

# %% [markdown]
#  # Hypothetical: 
#  After seeing your work, your stakeholders come to you and say that they can collect more data, but want your guidance before starting. How would you advise them based on your EDA and model results?

# %% [markdown]
# Based on the EDA and model results for Starcraft 2, here's how I would advise the stakeholders regarding collecting more data : I see that the variables that hold the most weight in predicting a player's rank are ActionLatency, APM, NumberOfPACs, GapBetweenPACs, & TotalHours. With this information some variables that I think are beneficial to gather: 
# 
# 1. Win Rate: The win rate of a player can be a strong predictor of their rank. Players with higher win rates are likely to have higher ranks. 
# 2. Game Duration: The average duration of the player's games can provide insights into their playstyle and strategy. Longer game durations may indeicate more strategic gameplay. 
# 3. Race: The race chosen by a player (Protoss, Terran, or Zerg) can also be a predictor of their rank. Different races have unique strengths and weaknesses, and players may have varying levels of proficiency with each race.
# 4.Experience: The number of games played or the player's experience level can be indicative of their skill and rank. More experienced players tend to have higher ranks.
# 5. Average Resource Collection Rate: The rate at which a player collects resources (minerals and gas) during the game can reflect their ability to efficiently manage their economy and build an army.
# 
#  The data provided does not contain enough data for grandmaster, although it is understandable that this rank does not have a lot of data since this league represents the top 200 players in each region, the data provided is not enough to get an accurate model. 
# 
# 


