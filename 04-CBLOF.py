import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn import __version__ as sklearn_version

'''
Explore the Data
'''
print(pd.__version__)
print(sklearn_version)
# https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls
df = pd.read_excel("Superstore.xls")
sns.set_theme()
sns.regplot(x="Sales", y="Profit", data=df)
sns.despine()
plt.show()

# 处理数据
n_points = df.shape[0]
x1 = df['Sales'].values
x2 = df['Profit'].values
x1 = x1 - x1.min()
x1 = x1 / x1.max()
x2 = x2 - x2.min()
x2 = x2 / x2.max()
X = [[a, b] for (a, b) in zip(x1, x2)]
X = np.array(X)

'''
KMeans
'''
km = KMeans(n_clusters=8)
km.fit(X)

cluster_sizes = []
for label in range(km.n_clusters):
    size = sum(km.labels_ == label)
    cluster_sizes.append(size)

plt.figure(figsize=(10, 8))
for label in range(km.n_clusters):
    plt.scatter(x1[km.labels_ == label], x2[km.labels_ == label], label=f'{label} size: {cluster_sizes[label]}')
plt.legend()
plt.show()

'''
Find Large and Small Clusters
'''
alpha = 0.9
beta = 5
n_points_in_large_clusters = int(n_points * alpha)
n_outliers = 100
df_cluster_sizes = pd.DataFrame()
df_cluster_sizes['cluster'] = list(range(8))
df_cluster_sizes['size'] = df_cluster_sizes['cluster'].apply(lambda c: cluster_sizes[c])
df_cluster_sizes.sort_values(by=['size'], ascending=False, inplace=True)
print(df_cluster_sizes)

# small_clusters = []
# large_clusters = []
# count = 0
# for _, row in df_cluster_sizes.iterrows():
#     count += row['size']
#     if count < n_outliers:
#         small_clusters.append(row['cluster'])
#     else:
#         large_clusters.append(row['cluster'])

large_clusters = []
small_clusters = []
found_b = False
count = 0
clusters = df_cluster_sizes['cluster'].values
n_clusters = len(clusters)
sizes = df_cluster_sizes['size'].values

for i in range(n_clusters):
    print(f"-----------iterration {i}--------------")
    satisfy_alpha = False
    satisfy_beta = False
    if found_b:
        small_clusters.append(clusters[i])
        continue

    count += sizes[i]
    print(count)
    if count > n_points_in_large_clusters:
        satisfy_alpha = True

    print(sizes[i] / sizes[i + 1])
    if i < n_clusters - 1 and sizes[i] / sizes[i + 1] > beta:
        print("beta")
        satisfy_beta = True

    print(satisfy_alpha, satisfy_beta)
    if satisfy_alpha and satisfy_beta:
        found_b = True

    large_clusters.append(clusters[i])

print(n_points_in_large_clusters)
print(small_clusters)
print(large_clusters)

large_cluster_centers = km.cluster_centers_[large_clusters]

# Plot large and small clusters
labels_in_large_clusters = np.zeros((n_points))
for large_cluster in large_clusters:
    labels_in_large_clusters[km.labels_ == large_cluster] = 1

n_points_in_small_clusters = sum(1 - labels_in_large_clusters)
print(f"we have {n_points_in_small_clusters:0.0f} points in small clusters")

plt.figure(figsize=(10, 8))
plt.scatter(x1[labels_in_large_clusters == 1], x2[labels_in_large_clusters == 1], label=f'large clusters')
plt.scatter(x1[labels_in_large_clusters == 0], x2[labels_in_large_clusters == 0], label=f'small clusters')
plt.legend()
plt.title("Points in large and small clusters")
plt.show()

'''
Factor:
这里我们要计算CBLOF因子，也就是一个点到最近的大簇的距离。
如果这个点是大簇里面的点，那么直接计算他到簇中心的距离即可。
如果这个点不是大簇点，那么就要分别计算其到所有大簇的距离，选最小的那个作为因子。
'''


def get_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def decision_function(X, labels):
    n = len(labels)
    distances = []
    for i in range(n):
        p = X[i]
        label = labels[i]
        if label in large_clusters:
            center = km.cluster_centers_[label]
            d = get_distance(p, center)
        else:
            d = None
            for center in large_cluster_centers:
                d_temp = get_distance(p, center)
                if d is None:
                    d = d_temp
                elif d_temp < d:
                    d = d_temp
        distances.append(d)
    distances = np.array(distances)
    return distances


distances = decision_function(X, km.labels_)

threshold = np.percentile(distances, 99)
print(f"threshold is {threshold}")

anomaly_labels = (distances > threshold) * 1
print(anomaly_labels)

'''
Plot
'''
# np.meshgrid 从坐标向量返回坐标矩阵。
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
# print(xx)
X_mash = np.c_[xx.ravel(), yy.ravel()]
labels_mash = km.predict(X_mash)
Z = decision_function(X_mash, labels_mash) * -1
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 8))
# np.linspace主要用来创建等差数列
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), -threshold, 7), cmap=plt.cm.Blues_r)
plt.contourf(xx, yy, Z, levels=[-threshold, Z.max(), ], colors='orange')
a = plt.contour(xx, yy, Z, levels=[-threshold], linewidths=2, colors='red')
b = plt.scatter(x1[anomaly_labels == 0], x2[anomaly_labels == 0], c='white', s=20, edgecolor='k', label='Normal')
c = plt.scatter(x1[anomaly_labels == 1], x2[anomaly_labels == 1], c='black', s=20, label='Anormaly')
plt.legend([a.collections[0], b, c], ['learned decision function', 'inliers', 'outliers'],
           prop=matplotlib.font_manager.FontProperties(size=20), loc='lower right')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.title('Cluster-based Local Outlier Factor (CBLOF)')
plt.show()
