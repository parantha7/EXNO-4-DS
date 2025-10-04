# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## Developed By : PARANTHAMAN S
## Reg_No       : 212224040232

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
<img width="1575" height="865" alt="image" src="https://github.com/user-attachments/assets/50491322-fea7-4009-914c-a70000fe66a8" />

```
data.isnull().sum()
```
<img width="791" height="597" alt="image" src="https://github.com/user-attachments/assets/8a07dfb1-10ed-45ec-a34a-095749c2b26a" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1601" height="731" alt="image" src="https://github.com/user-attachments/assets/42fe3287-1455-49b7-aad6-e46c569f2b0a" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1603" height="737" alt="image" src="https://github.com/user-attachments/assets/f7ed8736-a0d1-4d16-9a3d-f8f281c88dea" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="1351" height="464" alt="image" src="https://github.com/user-attachments/assets/a9109295-bdb3-4a12-b581-d782aed107bd" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="887" height="563" alt="image" src="https://github.com/user-attachments/assets/c8ad586d-9c7d-4087-aaa6-fe745cb71f46" />

```
data2
```
<img width="1554" height="524" alt="image" src="https://github.com/user-attachments/assets/37575fb3-3f07-46ed-9283-14182468bfff" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1593" height="553" alt="image" src="https://github.com/user-attachments/assets/9267865a-54bf-425d-9e12-3e3b4e356f81" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1609" height="128" alt="image" src="https://github.com/user-attachments/assets/584db353-58c8-4458-9d12-c4ada5d6135c" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1591" height="132" alt="image" src="https://github.com/user-attachments/assets/376dd466-ad8c-4062-a7df-03462f835329" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="1087" height="106" alt="image" src="https://github.com/user-attachments/assets/9e390bf1-3913-4a25-a654-bee076d586c6" />

```
x=new_data[features].values
print(x)
```
<img width="921" height="227" alt="image" src="https://github.com/user-attachments/assets/5e2b62dd-2f6e-441a-b6bb-c2633675c861" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
<img width="865" height="208" alt="image" src="https://github.com/user-attachments/assets/01fc4f08-6f8f-441d-9029-299c9e1b6afc" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="921" height="148" alt="image" src="https://github.com/user-attachments/assets/cb49df06-1b06-4e39-8e30-6f819a34d086" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="763" height="108" alt="image" src="https://github.com/user-attachments/assets/04bfb757-e615-409c-bea4-387216c27c1b" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="719" height="90" alt="image" src="https://github.com/user-attachments/assets/4b493923-7ccb-4e2a-abf4-5793d3282c7c" />

```
data.shape
```
<img width="337" height="87" alt="image" src="https://github.com/user-attachments/assets/9973a9c9-656c-4953-ac9d-3c72e410313c" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
```

```
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y.values.ravel())
selected_feature_indices=selector.get_support(indices=True)
```

```
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="957" height="520" alt="image" src="https://github.com/user-attachments/assets/cdb69da1-50e8-484f-9339-e40ebc93aef2" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="906" height="391" alt="image" src="https://github.com/user-attachments/assets/a6b23610-c1a8-4475-9d9a-76365d8f97a7" />

```
tips.time.unique()
```
<img width="559" height="100" alt="image" src="https://github.com/user-attachments/assets/e64d48b3-1b0c-4ba1-bb89-f6a81c5fbc80" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="633" height="172" alt="image" src="https://github.com/user-attachments/assets/e8cf4762-80f2-4111-b18c-422097ab53d4" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="702" height="146" alt="image" src="https://github.com/user-attachments/assets/4624048b-25f8-4ca9-a935-0498d9999cb5" />



# RESULT:

```
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
```
