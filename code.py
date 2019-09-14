# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here

df = pd.read_json(path, lines=True)
df.columns = df.columns.str.replace(" ","_")
missing_data = df.isna()
# print(missing_data.sum())
df.drop(columns=['waist', "shoe_width","shoe_size", "bust","user_name","review_text", "review_summary"], inplace=True)
X = df.drop(columns=['fit'])
y = df['fit']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=6)

# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
g_by_category = df.groupby('category')
cat_fit = g_by_category['fit'].value_counts()
cat_fit = cat_fit.unstack()
cat_fit.plot.bar()
# Code ends here


# --------------
# Code starts here
cat_len = g_by_category['length'].value_counts().unstack()
cat_len.plot.bar()
# Code ends here


# --------------
# Code starts here
def get_cms(x):
    if isinstance(x,float):
        return x
    if isinstance(x,str) and len(x) < 4:
        return int(x[0])* 30.48
    return int(x[0])* 30.48 + int(x[4:-2])* 2.54


X_train.height = X_train.height.apply(get_cms)
X_test.height = X_test.height.apply(get_cms)

# Code ends here


# --------------
# Code starts here
X_train.isna().sum()

X_train.dropna(subset=["quality","height","length"],inplace=True)
X_test.dropna(subset=["quality","height","length"],inplace=True)
# assuming y values are automatically removed
X_train.bra_size.fillna(X_train.bra_size.mean(),inplace=True)
X_train.hips.fillna(X_train.hips.mean(),inplace=True)
X_test.bra_size.fillna(X_test.bra_size.mean(),inplace=True)
X_test.hips.fillna(X_test.hips.mean(),inplace=True)
mode_1 = X_train.cup_size.mode()[0]
mode_2 = X_test.cup_size.mode()[0]
X_train.cup_size.fillna(mode_1,inplace=True)
X_test.cup_size.fillna(mode_2,inplace=True)

# Code ends here


# --------------
# Code starts here

X_train = pd.concat([X_train, pd.get_dummies(X_train[['length','cup_size','category']])], axis=1)
X_train.drop(columns=['length','cup_size','category'], inplace=True)

X_test = pd.concat([X_test, pd.get_dummies(X_test[['length','cup_size','category']])], axis=1)
X_test.drop(columns=['length','cup_size','category'], inplace=True)
# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here
model = DecisionTreeClassifier(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)
precision = precision_score(y_test,y_pred,average=None)

print(score)
print(precision)
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model = DecisionTreeClassifier(random_state=6)
grid = GridSearchCV(estimator=model, param_grid=parameters)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# Code ends here


