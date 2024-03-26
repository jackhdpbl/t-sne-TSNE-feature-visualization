from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def tSNE(feature,label,titlename,picname):  #feature为np.array格式,包含所有特征,label也为np.array格式,与feature顺序对应,titlename即绘图时的图标题,picname即存储图像的名字
    # 使用t-SNE算法进行降维
    X = feature #简化表示
    y = label  #简化表示
    tsne = TSNE(n_components=2, random_state=0)  # n_components代表降维后空间的维度,random_state表示迭代开始的初始化点
    X_tsne = tsne.fit_transform(X) #拟合降维
    # 可视化降维后的数据
    plt.figure(figsize=(8, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))  #这里可以自己根据标签数(label)设置颜色范围,这里为10
    plt.clim(-0.5, 9.5)
    plt.title(str(titlename))  #根据输入的名字参数调用
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2') #分别命名X轴和Y轴的名字
    plt.savefig('./'+ str(picname)+'.png')

# 生成特征和标签
np.random.seed(0)
num_samples = 1000
num_features = 20
num_classes = 5

# 随机特征
features = np.random.rand(num_samples, num_features)

# 样本标签
labels = np.random.randint(0, num_classes, num_samples)

# 示例图名
title_name = "t-SNE Visualization"
pic_name = "tSNE_plot"

# 测试函数
tSNE(features, labels, title_name, pic_name)