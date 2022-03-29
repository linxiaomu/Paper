from sklearn.cluster import KMeans
import numpy as np
class mykmeans(KMeans):
    def __init__(self,Kmeans):
        self.__init__(KMeans)
# X = np.array([[1, 2], [1.5, 1.8], [5, 8],
#               [8, 8], [1, 0.6], [9, 11]])
import random
if __name__ == '__main__':
    # print(random.randint(-200,200))

    # # 坐标向量
    # a = np.array([1, 2, 3])
    # # 坐标向量
    # b = np.array([7, 8])
    # # 从坐标向量中返回坐标矩阵
    # # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
    # res = np.meshgrid(a, b)
    # print(res)
    # # 返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]
    print(np.linspace(0, 1, 100))