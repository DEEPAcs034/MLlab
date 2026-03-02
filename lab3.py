import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris=load_iris()
data=iris.data
target=iris.target
label_names=iris.target_names
#converting into dataframes
iris_df=pd.DataFrame(data,columns=iris.feature_names)#columns are features
pca=PCA(n_components=2)
data_reduced=pca.fit_transform(data) #converting from 4 to 2 dimenstion fits all the data as oer required thats is 2
reduced_df=pd.DataFrame(data_reduced,columns=['PC1','PC2'])
reduced_df['target']=target
#plotting graph
plt.figure(figsize=(10,15))
colors=['r','g','b']
for i, target in enumerate(np.unique(target)):
  plt.scatter(reduced_df[reduced_df['target']==target]['PC1'],
              reduced_df[reduced_df['target']==target]['PC2'],
              c=colors[i],label=label_names[i])
plt.title('PCA on Iris Dataset')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid()
plt.show()
