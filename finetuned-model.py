import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
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
from keras.layers import Dense, Dropout,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import layers
from keras import regularizers
from keras import Input
from keras.optimizers import SGD,Adagrad
from keras.models import load_model,Model


path= '/Users/neerajamenon/Desktop/uni/ip/best_model-10000-all.h5'

pre_trained = load_model(path)

data = pd.read_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/breast/all.csv')

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

selected_features = feature_scores.head(10000)['Feature']
X_train_fs = X_train[selected_features]

sm = SMOTE(random_state = 42)
X_train_res, y_train_res= sm.fit_resample(X_train_fs, y_train.ravel())

input_layer = Input(X_train_res.shape[1])


# for layer in pre_trained.layers[-4:]:
#     layer.trainable = False

# # Freeze first x layers
# for layer in pre_trained.layers[:15]:
#     layer.trainable = False
# for layer in pre_trained.layers[15:]:
#     layer.trainable = True

# pre_trained.trainable = False
pre_trained(input_layer,training=False) 

for layer in pre_trained.layers:
    layer.trainable = True


x = Dense(64, activation='relu',name='3')(input_layer)
x = Dropout(0.5)(x)

pretrained_output = pre_trained(input_layer)

x = Dense(32, activation='relu',name= '2')(x)
x = Dropout(0.5)(x)
x = Dense(16, activation='relu',name='1')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='nadam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

  
mc = ModelCheckpoint('best_model-ft-all.h5', save_best_only=True, verbose=1)

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

plt.plot(epochs, val_loss_vals, 'orange', label='Validation loss')
plt.plot(epochs, loss_vals, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, val_acc, 'green', label='Validation accuracy')
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

np.max(val_acc)

pred =np.round(model.predict(X_test[selected_features]),0)

training_pred = np.round(model.predict(X_train_res),0)

print(confusion_matrix(y_test, pred)) 

cm = confusion_matrix(y_test, pred, labels=[1,0],normalize='true')

sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.1%', cmap='YlGnBu')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')


plt.show()
print(classification_report(y_test, pred))