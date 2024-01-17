import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr,spearmanr



data = pd.read_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/breast/all.csv')
df = pd.DataFrame(data)

df.dropna()

Y=df['OS_STATUS']

X=df.drop(['PATIENT_ID','OS_STATUS'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                   random_state=42)

sm = SMOTE(random_state = 42)
X_train_res, y_train_res= sm.fit_resample(X_train, y_train.ravel())



# alpha = [0.001, 0.01, 0.1, 1, 10]
# param = {'alpha': alphas}
# grid_search = GridSearchCV(lasso, param, cv=5)
# grid_search.fit(X_train_res, y_train_res)


model = Lasso(alpha = 0.001 )
model.fit(X_train_res,y_train_res)
# selector1 = SelectFromModel(lasso)
# selector.fit(X_train_res,y_train_res)
# X_selected1 = selector1.transform(X_train_res)

train_score_ls =model.score(X_train_res,y_train_res)
test_score_ls =model.score(X_test,y_test)

print("Number of selected features: ", sum(model.coef_ != 0))

# SVM

# model=SVC() 
# model.fit(X_train_res,y_train_res)
# y_pred=model.predict(X_test[selected_features])


# Naive Bayes

#  model = GaussianNB()
# model.fit(X_selected, y_train_res)

# preds = np.round(model.predict(X_test_selected),0)
# y_pred_train = np.round(model.predict(X_selected),0)
# y_pred = model.predict(X_test_selected)
# accuracy = model.score(X_test_selected, y_test)

# train_score_ls =model.score(X_selected,y_pred_train)
# test_score_ls =model.score((X_test_selected),y_test)

# print("Number of selected features: ", sum(lasso.coef_ != 0))


preds = np.round(model.predict(X_test),0)
preds

trainpreds = np.round(model.predict(X_train_res),0)

print(confusion_matrix(y_test, preds))

cm = confusion_matrix(y_test, preds, labels=[1,0],normalize='true')

sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.1%', cmap='YlGnBu')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

print(classification_report(y_test, preds))
plt.show()

