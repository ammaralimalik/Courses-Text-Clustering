import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import mglearn
import scipy.sparse as sp

data = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
schoolData = np.loadtxt("school_codes.txt", dtype="str", delimiter="\t", skiprows=1)
deptData = np.loadtxt("dept_codes.txt", dtype="str", delimiter="\t", skiprows=1)
prefixData = np.loadtxt("prefix_codes.txt", dtype="str", delimiter="\t", skiprows=1)

index = data[:,0]
textData = data[:,1]

vectorizer = CountVectorizer(max_features=1000,max_df=.2)
X = vectorizer.fit_transform(textData)

cleanSchoolData = schoolData[:,1]
cleanDeptData = deptData[:,1]
cleanPrefixData = prefixData[:,1]

Sscore = []
clusters = []
for i in range(2, 58):
    clusters.append(i)
    kmean = KMeans(n_clusters = i).fit(X)
    if(i == 3):
        kmeans3 = kmean
    elif(i == 33):
        kmeans33 = kmean
    elif(i == 57):
        kmeans57 = kmean
    Sscore.append(silhouette_score(X, kmean.labels_, metric='manhattan'))
  

    

plt.title("Silhoutte Scores for K values")
plt.xlabel("Clusters")
plt.ylabel("Scores")
plt.plot(clusters,Sscore)


ARIscore = {'3 Clusters': adjusted_rand_score(cleanSchoolData, kmeans3.labels_) , '33 Clusters': adjusted_rand_score(cleanDeptData, kmeans33.labels_), '57 Clusters': adjusted_rand_score(cleanPrefixData, kmeans57.labels_)}
ARIC = list(ARIscore.keys())
ARIval = list(ARIscore.values())
plt.title("Adjusted Rand Score")
plt.xlabel("Clusters")
plt.ylabel("ARI score")
plt.bar(ARIC,ARIval, color='red')

lda = LatentDirichletAllocation(n_components=20, learning_method="batch", max_iter=20,n_jobs=-1).fit(X)
