import matplotlib.pyplot as plt
import numpy
import sklearn
#image recognition
from sklearn import datasets
#literal datasets
from sklearn import svm
#support vector machine

digits = datasets.load_digits()
#DATASET

print(digits.data)
print(digits.target)
#print(digits.images[0])

model = svm.SVC(gamma=0.0001, C=100)
#print(len(digits.data))
X,y = digits.data[:-10], digits.target[:-10]
model.fit(X,y)

print('Prediction:',(model.predict(digits.data[[-5]]))[0])
#print(digits.data[:-5])
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()