import numpy as npimport pandas as pdfrom sklearn.cluster import KMeansfrom sklearn.feature_extraction.text import CountVectorizerdata = np.loadtxt("descriptions.txt", dtype="str", delimiter="\t", skiprows=1)index = data[:,0]textData = data[:,1]