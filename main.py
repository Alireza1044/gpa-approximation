import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd

input_path = 'data/Elearning-Data-cut.xls'
train_data_count = 500


def read_input():
    input = pd.read_excel(input_path, 0)
    input = input.drop(['STUDENTN'], axis=1)
    input = input.fillna(0)
    output_gpa = pd.read_excel(input_path, 1)
    output_fail = pd.read_excel(input_path, 2)[['OUT_IN']]
    return input, output_gpa, output_fail


input, output_gpa, output_fail = read_input()
input = input.to_numpy()
output_gpa = output_gpa.to_numpy()
output_fail = output_fail.to_numpy()
input2 = []
output_fail2 = []

for i in range(len(input)):
    if output_fail[i] == 0:
        input2.append(input[i])
        input2.append(input[i])
        output_fail2.append([output_fail[i][0]])
        output_fail2.append([output_fail[i][0]])
    else:
        input2.append(input[i])
        output_fail2.append([output_fail[i][0]])

input2 = np.array(input2)
output_fail2 = np.array(output_fail2)

X_train = input[:train_data_count]
X_train2 = input2[:train_data_count]

Y_train = output_gpa[:train_data_count]
Y_train2 = output_fail2[:train_data_count]

X_test = input[train_data_count:]
X_test2 = input2[train_data_count:]

Y_test = output_gpa[train_data_count:]
Y_test2 = output_fail2[train_data_count:]
Y_test22 = output_fail[train_data_count:]

# X_train = keras.utils.normalize(X_train)
# X_test = keras.utils.normalize(X_test)
# X_test2 = keras.utils.normalize(X_test2)
# X_train2 = keras.utils.normalize(X_train2)

model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='relu'))
sgd = keras.optimizers.sgd(lr=0.04, decay=1e-6, nesterov=True)
model.compile(optimizer='Nadam', loss='mse', metrics=['mse'])
model.fit(X_train, Y_train, shuffle=True, epochs=150)

print(model.evaluate(X_test, Y_test))

print(model.predict(X_test))

model2 = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.01))
model2.add(Dense(1, activation='sigmoid'))
sgd = keras.optimizers.sgd(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_train2, Y_train2, validation_data=(X_test2, Y_test2), shuffle=True, epochs=150)

model2.evaluate(X_test, Y_test22)

model2.predict(X_test)

print([1 if n > 0.5 else 0 for n in model2.predict(X_test)])
