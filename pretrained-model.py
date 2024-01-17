import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import LeakyReLU
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import layers
from keras import regularizers
from keras.optimizers import SGD,Adagrad

data = pd.read_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/all/concatenated.csv')
df = pd.DataFrame(data)


df.dropna()

Y=df['OS_STATUS']

X=df.drop(['PATIENT_ID','OS_STATUS'],axis=1)


print(df.info())

print(df['OS_STATUS'].value_counts())




X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                   random_state=42)

F_values, p_values = f_classif(X_train, y_train)

feature_scores = pd.DataFrame({'Feature': X.columns, 'F-value': F_values})

feature_scores = feature_scores.sort_values(by=['F-value'], ascending=False)

selected_features = feature_scores.head(20000)['Feature']

X_train_fs = X_train[selected_features]

print(X_train.shape,X_train_fs.shape,y_train.shape,y_train.ravel().shape)


sm = SMOTE(random_state = 42)
X_train_res, y_train_res= sm.fit_resample(X_train_fs, y_train.ravel())

model = Sequential()
model.add(layers.Dense(700,input_shape=(X_train_res.shape[1],),activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dense(500))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.Dense(350))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(200,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.Dense(25))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(5,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
model.add(layers.Dense(1,activation='sigmoid',kernel_initializer='ones',
                     kernel_regularizer=regularizers.L1(0.05),
                     activity_regularizer=regularizers.L1(0.05)))
model.summary() 
adagrad = Adagrad(learning_rate=0.001)
sgd = SGD(lr=0.001)
model.compile(optimizer='nadam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])
 
mc = ModelCheckpoint('best_model-20000-all.h5', save_best_only=True, verbose=1)

es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max',
                                   patience=10,
                                   restore_best_weights=True)

history = model.fit(X_train_res, 
                    y_train_res,
                    callbacks=[es,mc],
                    epochs=100, 
                    batch_size=10,
                    validation_split=0.3,
                    verbose=1)

history_dict = history.history

loss_vals = history_dict['loss'] 
val_loss_vals = history_dict['val_loss'] 

epochs = range(1, len(loss_vals) + 1) 

plt.plot(epochs, loss_vals, 'bo', label='Training loss')
plt.plot(epochs, val_loss_vals, 'orange', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'green', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

np.max(val_acc)

preds =np.round(model.predict(X_test[selected_features]),0)
preds

trainpreds = np.round(model.predict(X_train_res),0)

# confusion matrix
print(confusion_matrix(y_test, preds)) 

cm = confusion_matrix(y_test, preds, labels=[1,0],normalize='true')

sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.1%', cmap='YlGnBu')

plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix')


plt.show()
print(classification_report(y_test, preds))
