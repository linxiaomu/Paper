from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2], [1.5, 1.8], [5, 8],
              [8, 8], [1, 0.6], [9, 11]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
centers = kmeans.cluster_centers_  # 两组数据点的中心点
labels = kmeans.labels_  # 每个数据点所属分组
print(centers)
print(labels)

for i in range(len(labels)):
    plt.scatter(X[i][0], X[i][1], c=('r' if labels[i] == 0 else 'b'))
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)

predict = [[2,1], [6,9],[4,6]]
label = kmeans.predict(predict)
for i in range(len(label)):
    plt.scatter(predict[i][0], predict[i][1],
                c=('r' if label[i] == 0 else 'b'), marker='x')

plt.show()
