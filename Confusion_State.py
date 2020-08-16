import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from scipy import signal 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc, precision_score,recall_score,f1_score
from xgboost import XGBClassifier
import lightgbm as lgbm

path =r"D:\Master UNIVPM\Projects\03\EEG_data.csv"

Data = pd.read_csv(path)

plt.plot(Data["Raw"]) # We notice a problem in Subject 6

# Exploring Data
print(Data.dtypes)
print(Data.columns)
print("Data shape:",Data.shape)
print(Data.head())
print(Data.describe())
print(Data.info())
print(Data.isnull().sum())
print(Data['SubjectID'].value_counts())
print(Data['VideoID'].value_counts())
print(Data['predefinedlabel'].value_counts())
print(Data['user-definedlabeln'].value_counts())

# Data Cleaning
# Change a Typo error as the Dataset Creator stated in Source (Kaggle)
# Sub 3 Vid 5 
Data[4532:4655]['predefinedlabel']=1.0
# Exclude SubjectID=6.0. Aquesitioning Device was worn incorrectly during recording
Data=Data.drop(Data.index[list(range(7717, 8992))])

# convert into multiindexed Dataframe
Data.set_index(['SubjectID','VideoID'], inplace = True)

# Featuer Extracting
# Create Featers Dataframes
df_Means=pd.DataFrame()
df_std=pd.DataFrame()

# Applying Featuer Extracting
SubjectID = [0,1,2,3,4,5,7,8,9]
for i in SubjectID:
    for k in range (0,10):
        SubjectID_df =   Data.iloc[Data.index.get_level_values('SubjectID') == float(i)]
        Sample_df = SubjectID_df.iloc[SubjectID_df.index.get_level_values('VideoID') == float(k)]
        df_Means=pd.concat([df_Means,mean(Sample_df).to_frame()],axis=1)
        df_std=pd.concat([df_std,Sample_df.std().to_frame()],axis=1)     

# Setting Labels and Featuers
Desiered_Featuers_Columnes =  ['Attention', 'Mediation', 
                               'Raw', 'Delta', 'Theta', 
                               'Alpha1', 'Alpha2',
                               'Beta1', 'Beta2', 
                               'Gamma1', 'Gamma2']
Labels_Columens = ['user-definedlabeln']

df_Means=df_Means.transpose()
df_std=df_std.transpose()

Labels = df_Means[Labels_Columens]

df_Means=df_Means[Desiered_Featuers_Columnes]
df_Means.columns=['Attention_mean', 'Mediation_mean', 
                               'Raw_mean', 'Delta_mean', 'Theta_mean', 
                               'Alpha1_mean', 'Alpha2_mean',
                               'Beta1_mean', 'Beta2_mean', 
                               'Gamma1_mean', 'Gamma2_mean']
df_std=df_std[Desiered_Featuers_Columnes]
df_std.columns=['Attention_std', 'Mediation_std', 
                               'Raw_std', 'Delta_std', 'Theta_std', 
                               'Alpha1_std', 'Alpha2_std',
                               'Beta1_std', 'Beta2_std', 
                               'Gamma1_std', 'Gamma2_std']

df_Featuers = pd.concat([df_Means,df_std],axis=1)

Featuers_and_Labels =  pd.concat([df_Featuers,Labels],axis=1)

# Prepare Train and test Data
splitRatio = 0.3
train, test = train_test_split(Featuers_and_Labels ,test_size=splitRatio,
                               random_state = 123, shuffle = True)

train_X = train.drop(Labels_Columens, axis=1)
train_Y = train[Labels_Columens]


test_X = test.drop(Labels_Columens, axis=1)
test_Y = test[Labels_Columens]

#Buliding Model

Clf_XGBosster =  XGBClassifier(learning_rate =0.1,
                            n_estimators=1000,
                            max_depth=10,
                            min_child_weight=1,
                            gamma=0.015,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=0.005,
                            objective= 'binary:logistic',
                            nthread=4,
                            scale_pos_weight=1,
                            seed=27,
                            random_state=777)
# we need to use values.ravel to ensure the labels are sent to classifier correctly
Clf_XGBosster.fit(train_X,train_Y.values.ravel())
print(Clf_XGBosster)

pred_y_XGBoost = Clf_XGBosster.predict(test_X)


## Metrics
print("Accuracy:",accuracy_score(test_Y, pred_y_XGBoost))
print("f1 score:", f1_score(test_Y, pred_y_XGBoost))
print("confusion matrix:",confusion_matrix(test_Y, pred_y_XGBoost))
print("precision score:", precision_score(test_Y, pred_y_XGBoost))
print("recall score:", recall_score(test_Y, pred_y_XGBoost))
print("classification report:", classification_report(test_Y, pred_y_XGBoost))

## Plots
### 01 plot Confusion Matrix as heat map
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(test_Y, pred_y_XGBoost),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

### 02 plot ROC curve
predicting_probabilites_XGB = Clf_XGBosster.predict_proba(test_X)[:,1]
fpr,tpr,thresholds = roc_curve(test_Y,predicting_probabilites_XGB)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20) 

