from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn import svm
clf = svm.LinearSVC()
clf.probability=True
clf.fit(X_train, y_train)


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state = 1234)
lr_model.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
dt_model.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state = 0)
rf_model.fit(X_train,y_train)


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

