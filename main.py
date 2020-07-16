import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import pandas as pd

input_path = 'data/Elearning-Data-cut.xls'


def read_input():
    input = pd.read_excel(input_path, 0)
    input = input.drop(['STUDENTN'], axis=1)
    input = input.fillna(0)
    output_gpa = pd.read_excel(input_path, 1)
    output_fail = pd.read_excel(input_path, 2)[['OUT_IN']]
    return input, output_gpa, output_fail


def prepare_data(data):
    res = []
    x = data.iterrows()
    print(input)
    for name in data:
        print(list(name))


train_data_count = 500

if __name__ == '__main__':
    print("Hi")
    input, output_gpa, output_fail = read_input()
    input = input.to_numpy()
    output_gpa = output_gpa.to_numpy()
    output_fail = output_fail.to_numpy()
    X_train = input[:train_data_count]
    Y_train = output_gpa[:train_data_count]
    X_test = input[train_data_count:]
    Y_test = output_gpa[train_data_count:]

    X_train = keras.utils.normalize(X_train)
    X_test = keras.utils.normalize(X_test)

    if True:
        model = Sequential()
        # model.add(Input(27,(27,)))
        model.add(Dense(150, activation='relu'))
        model.add(Dropout(0.04))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.04))
        model.add(Dense(1, activation='relu'))
        sgd = keras.optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
        model.fit(X_train, Y_train, batch_size=15, epochs=250)
        model.evaluate(X_test, Y_test)

# student_no = data['STUDENTN']
# gender = data['GENDER']
# age = data['AGE']
# marriage_stat = data['MARRY']
# distance = data['FASELEH']
# prev_gpa = data['BEFORMEA']
# job_stat = data['SHAGHEL']
# in_tehran = data['TEHRAN']
# around_tehran = data['URBEN']
# out_tehran = data['SHARI']
# independant = data['HAZ_ME']
# dependant = data['HAZ_FAM']
# co_dependant = data['HAZI_MF']
# computer_assur = data['ETMINAN']
# exam_stress = data['EXAMANA']
# aggressiveness = data['LOCF']
# confidence = data['L_ATUNOM']
# internet_speed = data['SPEEDAB']
# laptop_access = data['LAPTOP']
# mobile = data['MOBILE']
# elec_habit = data['ELHABIT']
# first_elec_gpa = data['FIRST.SY']
# satisfaction = data['SATSFIY']
# time_manage = data['TIMENNG']
# self_manage = data['SELFREG']
# elec_appearance = data['HOZOR']
# elec_oppinion = data['ELATITI']
# busyness = data['ST_BUSY']
