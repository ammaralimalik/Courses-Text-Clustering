import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import seaborn as sns

data = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
schoolData = np.loadtxt("school_codes.txt", dtype="str", delimiter="\t", skiprows=1)
deptData = np.loadtxt("dept_codes.txt", dtype="str", delimiter="\t", skiprows=1)
prefixData = np.loadtxt("prefix_codes.txt", dtype="str", delimiter="\t", skiprows=1)

index = data[:,0]
textData = data[:,1]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textData)

kmean3 = KMeans(n_clusters=3,max_iter=100).fit(X)
kmean33 = KMeans(n_clusters=33,max_iter=100).fit(X)
kmean57 = KMeans(n_clusters=57,max_iter=100).fit(X)
