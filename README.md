# PythonConsensusClustering
## Author: Sander Siepel. Feel free to edit and use to your liking. 
A Python class for Consensus Clustering (with K-medoids and PAM). The code is based on the original paper by Stefano Monti, Pablo Tamayo, Jill Mesirov & Todd Golub. This paper can be found at: https://link-springer-com.vu-nl.idm.oclc.org/article/10.1023/A:1023949509487.

## Why use Consensus Clustering?
The following section is taken from my MS thesis:

When performing a cluster analysis, one often encounters two problems: the number of groups (clusters) is unknown, and their forms are unknown. These two unknown
properties define the most fundamental issues to be addressed when clustering data. To this end, Monti et al. developed a general, model-independent resampling-based
methodology of class discovery and clustering validation (and visualization) called Consensus Clustering (CC). In this methodology, the authors provide several ways to (i) determine the number of clusters and (ii) to assign confidence to the selected number of clusters.

The main principle underlying Consensus Clustering is simple: the more the found clusters are robust to sampling variability, the higher the confidence that these clusters
represent real structure. One can thus create several perturbations of the original data set, apply a clustering algorithm on it (such as k-medoids), and measure the agreement (or consensus) among the runs.

## How to use? 
```
# The code takes as input `data`, which is a distance/(dis)similarity matrix. For example, use the Gower's Distance for mixed data types.
# df is your pd.DataFrame, holding your items to be clustered
data_gower = pd.DataFrame(gower.gower_matrix(df))
c = ConsensusCluster(data=data_gower, resample_iterations=30, min_k = 2, max_k = 7, resample_fraction=0.8, res_folder_name=None, verbose=True)

c.plot_dist_hist()
c.plot_clustermaps()
c.plot_CDFs()
c.get_sub_clusters_scores(verbose=True)
```


