import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'E:\Ananthufiles\mnist-in-csv\mnist_train.csv')  #  Please change the file location
print(data.head())
a = data.iloc[3, 1:].values
a = a.reshape(28, 28).astype('uint8')
plt.imshow(a)

df_x = data.iloc[:, 1:]
df_y = data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

print(y_train.head())

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

pred = rf.predict(x_test)  # Prediction on test data
print(pred)

s = y_test.values

count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1
print(count)

k = (count/len(pred))*100
print("Percentage accuracy:", k, "%")
