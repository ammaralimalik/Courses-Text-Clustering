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
from sklearn.cluster import AgglomerativeClustering


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

kmeans3 = 0
kmeans33 = 0
kmeans57 = 0

def run_kmeans():
    Sscore = []
    clusters = []
    for i in range(3, 58):
        clusters.append(i)
        kmean = KMeans(n_clusters = i).fit(X)
        if(i == 3):
            kmeans3 = kmean
        elif(i == 33):
            kmeans33 = kmean
        elif(i == 57):
            kmeans57 = kmean
        Sscore.append(silhouette_score(X, kmean.labels_, metric='manhattan'))
        

    plot_silhoutteScore(clusters,Sscore)
    plot_ariScores(kmeans3.labels_,kmeans33.labels_,kmeans57.labels_)
    
    
def run_aggClustering():
    Sscore = []
    clusters = []
    for i in range(3,58):
        clusters.append(i)
        agg = AgglomerativeClustering(n_clusters = i).fit(X.toarray())
        Sscore.append(silhouette_score(X, agg.labels_, metric='manhattan'))
        if(i == 3):
            agg3 = agg
        elif(i == 33):
            agg33 = agg
        elif(i == 57):
            agg57 = agg
    
    plot_silhoutteScore(clusters, Sscore)
    plot_ariScores(agg3.labels_, agg33.labels_, agg57.labels_)
    
    
def plot_silhoutteScore(X, y):
    plt.title("Silhoutte Scores")
    plt.xlabel("Clusters")
    plt.ylabel("Scores")
    plt.plot(X,y,color='red')
    plt.show()

def plot_ariScores(label1,label2,label3):
    ARIscore = {'3 Clusters': adjusted_rand_score(cleanSchoolData, label1) , '33 Clusters': adjusted_rand_score(cleanDeptData, label2), '57 Clusters': adjusted_rand_score(cleanPrefixData, label3)}
    ARIC = list(ARIscore.keys())
    ARIval = list(ARIscore.values())
    plt.title("Adjusted Rand Score")
    plt.xlabel("Clusters")
    plt.ylabel("ARI score")
    plt.bar(ARIC,ARIval, color='blue')
    plt.show()
    
        

      
        
      
        
      
        