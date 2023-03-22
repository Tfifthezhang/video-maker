import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as LR

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return(xx, yy)


centers = [[-1, 1], [1, -1]]

X, y = datasets.make_blobs(n_samples=100, centers=centers, cluster_std=0.45, random_state=0)

xx,yy=make_meshgrid(X[:,0],X[:,1])

clf=LR()
	
#sns.set(style='white')

#plt.style.use('_mpl-gallery')
						   
clf.fit(X,y)
Z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,8),frameon=False)
plt.axhline(0, color='black', linewidth=2.5, alpha=0.8)
plt.axvline(0, color='black', linewidth=2.5, alpha=0.8)
# plt.set_xticks([])
# plt.set_yticks([])
plt.contourf(xx,yy,-Z,cmap=plt.cm.RdBu,alpha=0.8)
for i,v in [[1,'r'],[0,'b']]:
	plt.scatter(X[y==i][:,0],X[y==i][:,1],c=v, s=120,edgecolor='k')
plt.show()