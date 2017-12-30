import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
import pydotplus

iris = load_iris()

test_idx = [0,50,100]
##training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

##testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

##initialize classifier

clf = tree.DecisionTreeClassifier()
##fit the training set
clf.fit(train_data,train_target)
##make predictions
predictions = clf.predict(test_data)

##print classifier accuracy
print 'classifier accuracy is ', accuracy_score(predictions,test_target)*100, ' %'

dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                      class_names=iris.target_names,
                       filled=True, rounded=True, impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('myiris.pdf')
