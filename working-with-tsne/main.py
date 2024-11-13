import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
with open('E:\\Work\\University\\PR\\Acoustic_prosody_Example\\data-mixed.json') as user_file:
   data_json = json.load(user_file)
dataFrame=pd.DataFrame(data_json)
traget=dataFrame.label
del dataFrame['label']
X=dataFrame
X_Scaled=StandardScaler().fit_transform(X)
# X=X_Scaled
x_pac=PCA(0.95).fit_transform(X,traget)
X=x_pac
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X,y=traget)

# pca=PCA(n_components=2)
# X_pac= pca.fit_transform(X,y=traget)
# print('Explained variability per principal component: {}'.format(pca.explained_variance_ratio_))

resultDF = pd.DataFrame(data = X_tsne
             , columns = ['t-sne1', 't-sne2'])
resultDF['label']=traget
plos=sns.scatterplot(data=resultDF,x='t-sne1',y='t-sne2',hue='label',style='label')
plos.set_title('T-SNE of Gender voice Dataset')

plt.show()
