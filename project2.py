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
    print("Running KMeans()")

    Sscore = []
    clusters = []
    for i in range(3, 58):
        clusters.append(i)
        kmean = KMeans(n_clusters = i).fit(X)
        score = silhouette_score(X, kmean.labels_, metric='manhattan')
        Sscore.append(score)
        if(i == 3):
            kmeans3 = kmean
            print("Silhoutte Score at 3 clusters: ",  score)
        elif(i == 33):
            kmeans33 = kmean
            print("Silhoutte Score at 33 clusters: ",  score)
        elif(i == 57):
            kmeans57 = kmean
            print("Silhoutte Score at 57 clusters: ",  score)
        
        
        

    plot_silhoutteScore(clusters,Sscore)
    plot_ariScores(kmeans3.labels_,kmeans33.labels_,kmeans57.labels_)
    print()
    
    
def run_aggClustering():
    print("Running AgglomerativeClustering()")
    
    Sscore = []
    clusters = []
    for i in range(3,58):
        clusters.append(i)
        agg = AgglomerativeClustering(n_clusters = i).fit(X.toarray())
        score = silhouette_score(X, agg.labels_, metric='manhattan')
        Sscore.append(score)
        if(i == 3):
            agg3 = agg
            print("Silhoutte Score at 3 clusters: ",  score)
        elif(i == 33):
            agg33 = agg
            print("Silhoutte Score at 33 clusters: ",  score)
        elif(i == 57):
            agg57 = agg
            print("Silhoutte Score at 57 clusters: ",  score)
    
    plot_silhoutteScore(clusters, Sscore)
    plot_ariScores(agg3.labels_, agg33.labels_, agg57.labels_)
    print()
    
def run_ldaClustering():
    print("Running lda()")
    Sscore = []
    clusters = []
    for i in range(3,58):
        lda = LatentDirichletAllocation(n_components=i,learning_method='online').fit(X)
    print()
    
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
    
        
def run_clustering():
    run_kmeans()
    run_aggClustering()
    run_ldaClustering()
      
        
      
        
      
        