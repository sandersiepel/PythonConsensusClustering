class ConsensusCluster:
    def __init__(self, data, resample_iterations, min_k, max_k, resample_fraction, res_folder_name=None, verbose=False):
         """ A class to perform Consensus K-medoids (with Partitioning Around Medoids) Clustering, based on the original paper by Stefano Monti, Pablo Tamayo, Jill Mesirov & Todd Golub
             The paper can be found at: https://link-springer-com.vu-nl.idm.oclc.org/article/10.1023/A:1023949509487

        Attributes:
            data                   A distance or (dis)similarity matrix that holds values within [0, 1] for each pair of items i and j in the data set
                                   For example, use the Gower's Distance matrix as input for mixed data types
            resample_iterations    Number of iterations. Parameter H in the original algorithm
            min_k                  Lowest value of k to try
            max_k                  Highest value of k to try
            resample_fraction      Fraction of original data set for the perturbed data sets. These data sets are then size: N*resample_fraction.
            res_folder_name        Folder path to save the output
            verbose                Boolean that states whether or not to output loggings
        """
            
        self.sub_clusters = []
        self.hierarchical_plot = None
        self.cluster_assignments = []
        self.consensus_matrix = None       
        self.data = data
        self.resample_iterations = resample_iterations
        self.min_k = min_k
        self.max_k = max_k
        self.resample_fraction = 0.8
        self.iteration_tables = []
        self.consensus_matrices = []
        self.verbose = verbose
        self.res_folder_name = res_folder_name
        self.cluster()
        
    def cluster(self):
        start_clustering = time.perf_counter()
        for k in range(self.min_k, self.max_k+1):
            # Perform clustering for this value of k. This clustering happens 'resample_iterations' times
            if self.verbose: print(f"Clustering for k={k}")
            sub = SubClusters(resample_iterations=self.resample_iterations, data=self.data, k=k, resample_fraction=self.resample_fraction)
            self.sub_clusters.append(sub)  # To keep track of their individual (silhouette) scores
            
            # The 'iteration_table' is of size N * resample_iterations; it holds the cluster assignments for 
            # each patient for each iteration
            self.iteration_tables.append(sub.iteration_table)
        
        if self.verbose:
            print(f"Clustering took {round(time.perf_counter() - start_clustering, 1)} seconds")
        
        # Now that we have the iteration tables for all k-values, make a consensus table with all the iteration tables
        if self.verbose:
            print(f"Starting to make the consensus tables")
            
        c_table = self.make_consensus_tables()
        
        
    def make_consensus_tables(self):
        # Loop over the iteration tables (with the cluster assignments) and combine them into a consensus table of size N * N
        for i_table in self.iteration_tables:  # df; iteration table for each k; consists of patient ID index and sample columns
            # Make the square matrix for N x N
            c_matrix = np.zeros(shape=(len(i_table), len(i_table)))
            c_matrix[:] = np.NaN  # Replace with NaN to keep the diagonal NaN 
            iteration_table = i_table.to_numpy()
            
            # Find all i,j combinations of patients that need a consensus index value
            comb = list(combinations(list(range(0, iteration_table.shape[0])), 2))

            for c in comb:
                both_clustered = 0
                same_cluster = 0

                for i, j in zip(iteration_table[c[0]], iteration_table[c[1]]):
                    if i >= 0 and j >= 0:
                        both_clustered += 1

                        if i == j:
                            same_cluster += 1

                res = same_cluster/both_clustered if both_clustered != 0 else 0

                c_matrix[c[0]][c[1]] = res
                c_matrix[c[1]][c[0]] = res
            
            self.consensus_matrices.append(pd.DataFrame(c_matrix))
        
        # Return a dataframe of the consensus_matrix (size: N*N)
        return self.consensus_matrices
    
    def plot_dist_hist(self):
        for idx, c_matrix in enumerate(self.consensus_matrices):
            fig = sns.histplot(data=c_matrix.to_numpy(dtype='float').ravel(), kde=True, bins=10)
            plt.xlabel("Consensus index value")
            
            if self.res_folder_name:
                if not os.path.isdir(f"{self.res_folder_name}"):
                    os.makedirs(f"{self.res_folder_name}")
                
                plt.savefig(f"{self.res_folder_name}/k{idx+self.min_k}_disthist.png")
                
            plt.show()
            
    def plot_clustermaps(self):
        cmap = sns.light_palette("red", reverse=False, as_cmap=True)
        
        for idx, c_matrix in enumerate(self.consensus_matrices):
            plt.figure(idx)
            sns_plot = sns.clustermap(c_matrix.replace(np.nan, 1), cmap=cmap, method='average')
            
            if self.res_folder_name:
                if not os.path.isdir(f"{self.res_folder_name}"):
                    os.makedirs(f"{self.res_folder_name}")
                    
                sns_plot.savefig(f"{self.res_folder_name}/k{idx+self.min_k}_heatmap.png")
            
    def plot_CDFs(self):
        figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
        
        for k_value, _c_matrix in enumerate(self.consensus_matrices):
            flat_matrix = _c_matrix.copy()
            flat_matrix = flat_matrix.values.ravel()
            flat_matrix.sort()
                        
            plt.plot(flat_matrix, np.linspace(start=0, stop=1, num=len(flat_matrix)), label=f"k={2+k_value}")

        plt.xlabel('Consensus index value')
        plt.ylabel('CDF')
        plt.title(f'Consensus matrix CDF for k={self.min_k} to k={self.max_k}')
        plt.legend(loc=(1.04,0))
        
        if self.res_folder_name:
            if not os.path.isdir(f"{self.res_folder_name}"):
                os.makedirs(f"{self.res_folder_name}")
            
            plt.savefig(f"{self.res_folder_name}/CDF.png")
        
        plt.show()
            
    def get_sub_clusters_scores(self, verbose=False):
        # Returns the mean silhouette score for each value of k. 
        # Mean is calculated over 'resample_iterations' individual silhouette scores.
        if verbose:
            for idx, sub_cluster in enumerate(self.sub_clusters):
                mean_sil_score = np.mean(sub_cluster.silhouette_scores)
                print(f"K={self.min_k+idx}, mean silhouette score: {mean_sil_score}")
        else:
            return [np.mean(sub_cluster.silhouette_scores) for sub_cluster in self.sub_clusters] 
        
    def get_sub_clusters_inertia_scores(self, verbose=False):
        # Returns the mean inertia score for each value of k. 
        # Mean is calculated over 'resample_iterations' individual inertia scores.
        if verbose:
            for idx, sub_cluster in enumerate(self.sub_clusters):
                mean_inertia_score = np.mean(sub_cluster.inertia_scores)
                print(f"K={self.min_k+idx}, mean inertia score: {mean_inertia_score}")
        else:
            return [np.mean(sub_cluster.inertia_scores) for sub_cluster in self.sub_clusters] 


class SubClusters():
    """ This class is called by the ConsensusCluster class, and performs the actual clustering for each value of k. """
    
    def __init__(self, resample_iterations, data, k, resample_fraction):
        self.silhouette_score = None
        self.resample_iterations = resample_iterations
        self.data = data  # Entire dataframe of all episode data
        self.k = k
        self.resample_fraction = resample_fraction
        self.silhouette_scores = []
        self.inertia_scores = []
        
        # The 'iteration_table' is of size N * 'resample_iteartions' and holds for each patient, for each resample, the
        # patients assignment to one of the clusters. 
        self.iteration_table = None
        self.cluster()
        
    def cluster(self):
        iteration_table = pd.DataFrame(index=self.data.index)
        
        for h in range(self.resample_iterations):
            sample = self.resample(self.data)  # Take subset of the data
            column_name = f"iteration{h+1}"
            iteration_table[column_name] = [np.nan] * len(self.data)
            
            # Perform clustering on the sub_sample
            kmedoids = KMedoids(metric='precomputed', method="pam", init='k-medoids++', n_clusters=self.k, random_state=21).fit(sample)
            self.silhouette_scores.append(silhouette_score(X=sample, labels=kmedoids.labels_, metric="precomputed"))
            self.inertia_scores.append(kmedoids.inertia_)
            
            # Determine which items were in the sub-sample and fill in their cluster assignments in the iteration_table
            for idx, l in enumerate(kmedoids.labels_):
                corresponding_row_id = sample.index.values.tolist()[idx]  # Find the original ID
                iteration_table.at[corresponding_row_id, column_name] = int(l)  # And replace the value of that cell
    
        self.iteration_table = iteration_table
        return self.iteration_table
    
    def resample(self, df):
        # This function makes a subset of the data, based on the resample_fraction (e.g. returns 80% of data)
        
        # Amount of row-column pairs to delete to obtain a new df with fraction size relative to original df. 
        to_delete = df.shape[0] - int((self.resample_fraction * df.shape[0]))

        # Take all of df indices and randomly select to_delete amount (30). These are the indices we delete to create our sub-set. 
        indices_to_drop = random.sample(df.index.values.tolist(), to_delete)

        # Create our new df that has size len(original df) * fraction
        subset = df.drop(indices_to_drop, axis=1).drop(indices_to_drop)
        return subset
    
