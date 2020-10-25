import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class KMeans():

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # create initial clusters based on K
        self.clusters = [[] for _ in range(self.K)]
        # initial centroids
        self.centroids = []

    def fit(self, X_train):
        """Summary or Description of the Function

            Parameters:
            X_train ([np.array,]): X input for training

            Returns:
            None

        """
        self.X_train = X_train
        self.n_samples, self.n_features = X_train.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X_train[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # create clusters for each iteration
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()

            # prepare old and new calculated centroids
            centroids_old = self.centroids
            self.centroids = self._update_centroids(self.clusters)
            
            # check if converges
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()
        self.plot()

    def predict(self, X_test):
        """Summary or Description of the Function

            Parameters:
            X_test ([np.array,]): X input for prediction

            Returns:
            [cluster index]: cluster indices in a list

        """
        return self._get_cluster_labels(X_test)


    def _get_cluster_labels(self, X_test):
        # each sample will get the label of the cluster it was assigned to
        n_samples, features = X_test.shape

        # get empty labels
        labels = np.empty(n_samples)
   
        # we calculate closest centroid and append the centroid idx to labels
        for sample_idx, sample in enumerate(X_test):
            centroid_idx = self._get_nearest_centroid(sample, self.centroids)
            labels[sample_idx] = centroid_idx
        return labels

    def _create_clusters(self, centroids):
        # first, create an empty cluster list of lists based on K
        clusters = [[] for _ in range(self.K)]

        # loop over X_train
        for idx, sample in enumerate(self.X_train):
            centroid_idx = self._get_nearest_centroid(sample, centroids)
            clusters[centroid_idx].append(sample)
        # print('created clusters in _create_clusters', clusters)
        return clusters

    def _get_nearest_centroid(self, sample, centroids):
        # first get distances for given sample from each centroid
        distances = [self._get_distance(sample, point) for point in centroids]
        
        # get centroid index where distance is the smallest
        closest_index = np.argmin(distances)
        return closest_index

    def _update_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(cluster, axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_current):
        # distances between each old and new centroids, fol all centroids
        distances = [self._get_distance(centroids_old[i], centroids_current[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, cluster in enumerate(self.clusters):
            ax.scatter(*zip(*cluster))

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        ax.set_title('Own Implementation of K-Means Clustering')
        plt.show()