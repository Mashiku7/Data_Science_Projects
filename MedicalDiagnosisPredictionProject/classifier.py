from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import ensemble
import numpy as np



#List of classifiers used in the medicaldiagnosis classifier
clf = tree.DecisionTreeClassifier()

clf1 = svm.SVC()

clf2 = GaussianNB()

clf3 = ensemble.AdaBoostClassifier()



#Hematology Anthropometric parameter varible Data [BMI, Height, Weight]
x = [[20, 113, 237],
 [27, 89, 310], 
 [31,127, 295],
[23,65,112],
[24,71,	136],
[31,69,153],
[21,68,142],
[27,67,	144],
[17,68,123],
[31,69,141],
[30,70,136],
[27,67,112]]


BMI = x[0]
Height = x[1]
Weight = x[2]


y= ['chronic obstructive pulmonary disease','Obesity',
'TYPE 2 DIABETES MELLITUS AND HYPOCHROMIA', 'chronic obstructive pulmonary disease',
'chronic obstructive pulmonary disease','Obesity', 'TYPE 2 DIABETES MELLITUS AND HYPOCHROMIA',
'chronic obstructive pulmonary disease', 'chronic obstructive pulmonary disease',
'Obesity', 'TYPE 2 DIABETES MELLITUS AND HYPOCHROMIA',
 'Obesity']

# CHALLENGE - ...and train them on our data

clf = clf.fit(x, y)

clf1 = clf1.fit(x, y)

clf2 = clf2.fit(x, y)

clf3 = clf3.fit(x, y)

prediction = clf.predict([[21,127, 277]])
prediction1 = clf1.predict([[21,127, 277]])
prediction2 = clf2.predict([[21,127, 277]])
prediction3 = clf3.predict([[21,127, 277]])


# CHALLENGE compare their reusults and print the best one!

print(prediction)
print(prediction1)
print(prediction2)
print(prediction3)