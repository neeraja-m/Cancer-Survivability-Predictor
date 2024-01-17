
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import mean_squared_error
import category_encoders as cat_encoder
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn import preprocessing as prp
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest,f_classif
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr

# Pairwise correlation between final datasets
# all_related_data = pd.read_csv("/Users/neerajamenon/Desktop/uni/ip/model/datasets/all/concatenated.csv")
all_related_data = pd.read_csv("/Users/neerajamenon/Desktop/uni/ip/model/datasets/all-related/concatenated.csv")
# all_unrelated_data = pd.read_csv("/Users/neerajamenon/Desktop/uni/ip/model/datasets/all-unrelated/concatentated.csv")
breast_data = pd.read_csv("/Users/neerajamenon/Desktop/uni/ip/model/datasets/breast/all.csv")


# all_data_df = pd.DataFrame(all_data)
all_related_df=pd.DataFrame(all_related_data)
# all_unrelated_df=pd.DataFrame(all_unrelated_data)
breast_df=pd.DataFrame(breast_data)


Y1=breast_df['OS_STATUS']
X1=breast_df.drop(['PATIENT_ID','OS_STATUS'],axis=1)

Y2=all_related_df['OS_STATUS']
X2=all_related_df.drop(['PATIENT_ID','OS_STATUS'],axis=1)


F_values1, p_values1 = f_classif(X1, Y1)

feature_scores1 = pd.DataFrame({'Feature': X1.columns, 'F-value': F_values1})

feature_scores1 = feature_scores1.sort_values(by=['F-value'], ascending=False)

selected_features1 = feature_scores1.head(50)['Feature']

X1_selected = X1[selected_features1]
X1_selected = X1_selected.add_prefix('A_')

# ###############

F_values2, p_values2 = f_classif(X2, Y2)

feature_scores2 = pd.DataFrame({'Feature': X2.columns, 'F-value': F_values2})

feature_scores2 = feature_scores2.sort_values(by=['F-value'], ascending=False)

selected_features2 = feature_scores2.head(50)['Feature']

X2_selected = X2[selected_features2]
X2_selected = X2_selected.add_prefix('B_')


################
# Create heatmaps

combined_sf = pd.concat([pd.DataFrame(X1_selected), pd.DataFrame(X2_selected)], axis=1)

c_matrix = combined_sf.corr()

sns.heatmap(c_matrix, cmap='coolwarm', annot=True,vmin=-1, vmax=1)
plt.figure(figsize=(70,70))

plt.title('Feature Selection Correlation - Breast Cancer vs. Related Cancers ')
plt.savefig('/Users/neerajamenon/Desktop/uni/ip/model/datasets/all-related/fs-corr1.png')

###########################################################

# Compute Spearmans correlation
X1_selected = X1_selected.head(2999)
X2_selected = X2_selected.head(2999)

selected_df1 = X1_selected.dropna().loc[:, X1_selected.apply(pd.Series.nunique) != 1]
selected_df2 = X2_selected.dropna().loc[:, X2_selected.apply(pd.Series.nunique) != 1]
concatenated = pd.concat([selected_df1, selected_df2], axis=1)

c_matrix, p = spearmanr(concatenated)

mean_corr = np.mean(c_matrix)

print("Overall correlation value: ", mean_corr)
print("P value: ", p.mean())

###################
# Survived vs Deceased Heatmaps

all_data = pd.read_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/all/concatenated.csv')
all_df = pd.DataFrame(all_data)

y=all_df['OS_STATUS']
X=all_df.drop(['PATIENT_ID','OS_STATUS','OS_MONTHS'],axis=1)


F_values, p_values = f_classif(X, y)

feature_scores = pd.DataFrame({'Feature': X.columns, 'F-value': F_values})

feature_scores = feature_scores.sort_values(by=['F-value'], ascending=False)

selected_features = feature_scores.head(500)['Feature']


plt.figure(figsize=(60, 70))


df_survived = all_df[all_df['OS_STATUS'] == 1]

sns.heatmap(df_survived[selected_features], cmap='coolwarm',vmin=-1.5, vmax=1.5,yticklabels=False)
plt.xlabel('Genes')
plt.ylabel('Samples')
plt.title('Gene Expression Heatmap For Deceased Patients')
plt.savefig('/Users/neerajamenon/Desktop/uni/ip/model/datasets/all/heatmap500-1.png')


#######
