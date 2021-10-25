import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from keras import layers
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

#1. pobranie danych i preprocessing
dane = pd.read_csv("dane/winequality-red.csv",sep = ";")
# odrzucenie outlierow
dane = dane[dane["total sulfur dioxide"]<250]
#pogrupowanie ocen wina w ramach dwoch grup
groups = []
for i in dane["quality"]:
    if i<=6.5:
        groups.append(1)
    else:
        groups.append(2)

dane["quality_category"] = groups

#podzial na zmienne objasniajace X i zmienna docelowa Y
X = dane[dane.columns[0:11]]
Y = dane["quality_category"]

#podzial zbioru na train i test
X_train_all, X_test, Y_train_all, Y_test = train_test_split(X,Y, test_size = 0.2)

#podzial na walidacyjny
X_train_reduced, X_val, Y_train_reduced, Y_val = train_test_split(X_train_all,Y_train_all, test_size = 0.1)

#przeskalowanie danych dla calego zbioru treningowego
scaler = StandardScaler().fit(X_train_all)
X_train_all_scaled = scaler.transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

## oraz dla wydzielonego zbioru treningowego i walidacyjnego w stosunku do cale zbioru treningowego
X_val_scaled = scaler.transform(X_val)
X_train_reduced_scaled = scaler.transform(X_train_reduced)

#PCA na przeskalowanych danych na calym zbiorze treningowym
PCA_results = PCA(n_components = 6)

PCA_results.fit(X_train_all_scaled)
X_train_all_scaled_PCA = PCA_results.transform(X_train_all_scaled)
X_test_scaled_PCA = PCA_results.transform(X_test_scaled)## oraz dla wydzielonego zbioru treningowego i walidacyjnego w stosunku do cale zbioru treningowego
X_train_reduced_scaled_PCA = PCA_results.transform(X_train_reduced_scaled)
X_val_scaled_PCA = PCA_results.transform(X_val_scaled)

lista_zmiennych = ["X","Y","X_train_all", "X_test", "Y_train_all", "Y_test","X_train_reduced", "X_val", "Y_train_reduced", "Y_val", "X_train_all_scaled","X_test_scaled","X_val_scaled","X_train_reduced_scaled","X_train_all_scaled_PCA","X_test_scaled_PCA","X_train_reduced_scaled_PCA","X_val_scaled_PCA"]

print("lista zmiennych: \n" ,lista_zmiennych)


#dopiszmy jeszcze dane potrzebne do odpalenia sieci neuronowej

# skorzystajmy z dostepnej metody z dedykowanej biblioteki
smote = SMOTE(sampling_strategy = 'minority',random_state=1)
x_sm, y_sm = smote.fit_resample(X,Y)


# podzielmy jeszcze raz zbior na zbior treningowy i testowy
X_train1, X_test1, Y_train1,Y_test1 = train_test_split(x_sm,y_sm, test_size = 0.2)



# Przetworzmy dane do formatu wejsciowego do modelu

X_train_array_d = X_train1.values
X_test_array_d = X_test1.values


# transformujemy Y_test_duplicated i Y_test_duplicated

Y_train_d = np.array(Y_train1)
Y_test_d = np.array(Y_test1)

Y_test_d = OneHotEncoder().fit_transform(Y_test_d.reshape(-1,1)).toarray()
# zmienmy jeszcze format
Y_test_d = Y_test_d.astype(np.float32)

# transformujemy y_train
Y_train_d = OneHotEncoder().fit_transform(Y_train_d.reshape(-1,1)).toarray()
# zmienmy jeszcze format
Y_train_d = Y_train_d.astype(np.float32)

     