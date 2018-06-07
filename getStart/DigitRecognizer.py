import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('data/DigitRecognizer/train.csv')
test = pd.read_csv("data/DigitRecognizer/test.csv")

train_np = train.as_matrix()
test_np = test.as_matrix()


train_x = train_np[:,1:]
train_y = train_np[:,0]
test_x = test_np

pca = PCA(n_components=0.8)
train_x = pca.fit_transform(train_x)
test_np = pca.transform(test_np)

kNN = KNeighborsClassifier(n_neighbors=4)
kNN.fit(train_x, train_y)

predictions = kNN.predict(test_np)
df = pd.DataFrame({'ImageId':range(1, len(predictions)+1), 'Label':predictions})
df.to_csv('output/DigitRecognizer/DigitRecognizer_knn.csv', index=False)
