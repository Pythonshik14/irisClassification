import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

path = "/content/gdrive/My Drive/Home_works/iris.csv"
data = pd.read_csv(path)

params = data.drop("name", axis=1).corr().columns

x_train, x_test, y_train, y_test = train_test_split(data[params], data['name'], test_size=0.2, random_state=0)


gamma = [0.001, 0.01, 0.005, 0.02]
C = [1, 10, 50, 100, 200, 500, 1000]
parameters1 = {"gamma":gamma, 
               "C":C}

model = svm.SVC(**parameters1)
grid = GridSearchCV(model, parameters1, cv=4)
grid.fit(x_train, y_train)

pred = grid.predict(x_test)

print(list(pred) == y_test)