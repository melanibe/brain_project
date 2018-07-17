import numpy as np
import os
import pandas as pd 
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns

"""  Draw the cluster frequency plot by sleep stages. To see if the clusters that K-means discovers somehow corresponds
to the sleep stages.
Answer is NO
"""

############# INTRO ###############
# Read parameters
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="choose the feature matrix")
parser.add_argument("-y", "--y", help="choose y")
args = parser.parse_args()

type_agg = args.type

if args.y:
    y = args.y
else:
    y = 'y'

# Change directory if necessary
try: #for local run
    os.chdir("/Users/melaniebernhardt/Documents/RESEARCH PROJECT/")
    cwd = os.getcwd()
except: #for cluster run
    cwd = os.getcwd()



############ LOADING DATA #############
X = np.load(cwd+"/{}.npy".format(type_agg))
Y = np.load(cwd+"/{}.npy".format(y))
n,m = np.shape(X)


############# Draw the cluster frequency plot by sleep stages #############
f, ax = plt.subplots(3, sharex=False, figsize=(6,9))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.suptitle('Plots for feature matrix {}'.format(type_agg), fontsize=16)
for i,k in enumerate([2,6,20]):
    print("Beginning K-means")
    kmean = MiniBatchKMeans(n_clusters=k)
    clusters = kmean.fit_predict(X).reshape(-1,1)
    df = pd.DataFrame()
    df["cluster"]=clusters.reshape(-1).astype(int)
    df["sleep_stages"]=Y.reshape(-1).astype(int)
    l = df.groupby('cluster')['sleep_stages'].value_counts().unstack().fillna(0)  #groupby count and then plot it 
    print(l)
    subf = sns.countplot(x="cluster", hue="sleep_stages", data=df, ax = ax[i])
    subf.legend(fontsize=5, loc='upper left')
    subf.set_title("Frequency of each cluster per sleep stage for {} clusters".format(k),fontsize=9)
plt.show()


""" DEPRECATED. 
TRIED OTHER CLUSTERING BUT EITHER SLOW OR REALLY BAD.

from sklearn.cluster import AffinityPropagation
clf = AffinityPropagation(verbose=True)
clusters = clf.fit_predict(X)
df = pd.DataFrame()
df["cluster"]=clusters.reshape(-1).astype(int)
df["sleep_stages"]=Y.reshape(-1).astype(int)
l = df.groupby('cluster')['sleep_stages'].value_counts().unstack().fillna(0)  #groupby count and then plot it 
print(l)
sns.countplot(x="cluster", hue="sleep_stages", data=df)
plt.show()
from sklearn.cluster import SpectralClustering
clf = SpectralClustering(n_clusters=6)
clusters = clf.fit_predict(X)
df = pd.DataFrame()
df["cluster"]=clusters.reshape(-1).astype(int)
df["sleep_stages"]=Y.reshape(-1).astype(int)
l = df.groupby('cluster')['sleep_stages'].value_counts().unstack().fillna(0)  #gro
sns.countplot(x="cluster", hue="sleep_stages", data=df)
print(l)
plt.show()



from sklearn.mixture import GaussianMixture
clf = GaussianMixture(n_components=6)
clf.fit(X)
clusters = clf.predict(X)
df = pd.DataFrame()
df["cluster"]=clusters.reshape(-1).astype(int)
df["sleep_stages"]=Y.reshape(-1).astype(int)
l = df.groupby('cluster')['sleep_stages'].value_counts().unstack().fillna(0)  #groupby count and then plot it 
print(l)
sns.countplot(x="cluster", hue="sleep_stages", data=df)
plt.show()
"""