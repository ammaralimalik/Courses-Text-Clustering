import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
import mglearn
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering


#Load data into variables
data = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
schoolData = np.loadtxt("school_codes.txt", dtype="str", delimiter="\t", skiprows=1)
deptData = np.loadtxt("dept_codes.txt", dtype="str", delimiter="\t", skiprows=1)
prefixData = np.loadtxt("prefix_codes.txt", dtype="str", delimiter="\t", skiprows=1)

#Seperate the index from the text data
index = data[:,0]
textData = data[:,1]

#Use count vectorizer to transform data
vectorizer = CountVectorizer(max_features=1000,max_df=.2)
X = vectorizer.fit_transform(textData)

#Clean ground truth data
cleanSchoolData = schoolData[:,1]
cleanDeptData = deptData[:,1]
cleanPrefixData = prefixData[:,1]

print()
print("Type: run_clustering() to start program (takes a while)")

#Runs KMeans method and returns nothing
def run_kmeans():
    print("Running KMeans()")

    Sscore = []
    clusters = []
    counter = []
    inertias = []
    
    for i in range(3, 59):
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
    
        inertias.append(kmean.inertia_)
        counter.append(i)
        
    plot_kmeansElbow(counter, inertias)
    plot_silhoutteScore(clusters,Sscore)
    plot_ariScores(kmeans3.labels_,kmeans33.labels_,kmeans57.labels_)
    print()
    
  #Runs AgglomerativeClustering method and returns nothing
def run_aggClustering():
    print("Running AgglomerativeClustering()")
    
    Sscore = []
    clusters = []
    for i in range(3,59):
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
    
    
#Helper function to plot visualizations for scores
def plot_silhoutteScore(X, y):
    plt.title("Silhoutte Scores")
    plt.xlabel("Clusters")
    plt.ylabel("Scores")
    plt.plot(X,y,color='red')
    plt.show()
    
#Helper function to plot bar plots for scores
def plot_ariScores(label1,label2,label3):
    ari3 = adjusted_rand_score(cleanSchoolData, label1)
    ari33 = adjusted_rand_score(cleanDeptData, label2)
    ari57 = adjusted_rand_score(cleanPrefixData, label3)
    print("ARI Score at 3 clusters: ", ari3)
    print("ARI Score at 33 clusters: ", ari33)
    print("ARI Score at 57 clusters: ", ari57)
    print()
    ARIscore = {'3 Clusters': ari3 , '33 Clusters': ari33, '57 Clusters': ari57}
    ARIC = list(ARIscore.keys())
    ARIval = list(ARIscore.values())
    plt.title("Adjusted Rand Score")
    plt.xlabel("Clusters")
    plt.ylabel("ARI score")
    plt.bar(ARIC,ARIval, color='blue')
    plt.show()

#Helper function to plot line graph for optimal clusters
def plot_kmeansElbow(k, inertias):
    plt.plot(k, inertias, 'bx-',color='black')
    plt.title("Elbow Plot")
    plt.xlabel("Clusters")
    plt.ylabel("Distortions")
    plt.xlim(0,25)
    plt.show()

 #Runs clustering    
def run_clustering():
    run_kmeans()
    run_aggClustering()

    
   
      
        
      
        
      
        
