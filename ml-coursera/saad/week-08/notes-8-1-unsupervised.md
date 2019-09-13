# Coursera - Machine Learning - Week 8

# 8.1 - Unsupervised Learning

## Unsupervised Learning
- Learning from unlabeled data as opposed to labeled data. 
- A typical supervised learning problem is to find a decision boundary that separates one labeled class of examples from another.
- Unsupervised learning: data has no labels, so the dataset is `x1, x2, x3,...,xm`.
- This unlabeled training set is given to an algorithm and we ask the algorithm to find some structure in the data.
- An algorithm that groups similar examples into clusters are called clustering algorithms.
	- First example of an unsupervised learning algorithm.

## What is clustering good for?
- Market segmentation
	- Have a set of customers and want to group them into different categories so we can market to them better.
- Social Network analysis
	- Finding groups of related people in a database of users based on their interests/interactions.
- Organizing Computer Clusters
	- To reorganize resources and design its layout/network.
- Astronomoial Data Analysis
	- Galaxy formation data

## K-Means Algorithm
- We are given an unlabeled dataset and would like an algorithm to automatically group the data into coherent subsets or clusters for us.
- K-Means algo is by far the most widely used algorithm.
- Suppose we want to group the data into two clusters.
- Randomly initialize two points called the **cluster centroids** (one per cluster)
- An iterative algorithm where each iteration has two steps 
	1. Cluster Assignment
		- Parse each sample in the dataset and assign it to cluster 1 or 2 based on how 'close' that sample is to the centroid `c1` or `c2`.
	2. Centroid Movement
		- Look at all the points that have been assigned to a cluster, calculate their `mean position` and move the cnetroid there.
	Repeat these steps until there is no more change in the points assigned to each cluster.

### Algorithm Definition
- Input
	- `K` - the number of clusters we want to find in the data
		- There exists a process that helps us choose an optimal number of clusters. 
	- Training set `{x1, x2, x3, ..., xm}`
- Each training example `xi` is an `n` dimensional vector, not an `n + 1` dimensional vector.
	- Dropping the `x_0 = 1` convention.
- Randomly initialize `K` cluster centroids `mu1, mu2, mu3,..., muK` all of which are `n` dimensional.
- Repeat
	- **Cluster Assignment**
		For `i` = 1 to `m`
			`c(i)` := index (from 1 to `K`) of cluster centroid closest to `x(i)`.
			// This is a number corresponding to the closest cluster centroid to this the datapoint `x_i`.
			// It computes `||x_i - mu_k||^2` for all `k` in range(1, K)
			// Square distance is used as a convention, although minimising distance would also give the same value of `k`.
	- **Centroid Movement**
		For k = 1 to `K`
			`mu_k` := average (mean) of points assigned to cluster `k`. 
			- `mu_k` will still be the `n` dimensional vector.
		- What if the cluster has no points assigned to it? 
			- Eliminate the cluster so you will have `k - 1` clusters.
			- Randomly reinitialize the cluster centroid.

### K-Means and Non-separated Clusters
- So far we have applied K-means to a dataset with three very well-separated clusters.
- E.g. t-shirt sizing data (weight against height) is a positively correlated trend, and the data is not necessarily well separated.
- Unsupervised problem: which weight/height thresholds to use for small/medium/large sizes?
- Design a t-shirt that fits weight/height data for each of the three clusters: an example of market segmentation.
 
## Optimization Objective
- All the algorithms we have seen so far have had an optimization objective or some cost function that we are trying to minimise.
- K-Means also has an objective/cost function, even though it is isn't a supervised learning algo.
	- Will help us debug the algorithm and ensure it is running correctly.
	- Will also help us to avoid local optima.
- Two sets of variables being tracked in K Means
	- `c(i)` - index of cluster to which example `x(i)` is currently assigned.
	- `mu_k` the cluster centroid `k` - an `n` dimensional real vector.
	- `mu_c(i)` - the cluster centroid to which **example `x(i)`** has been assigned.
			- So if the sample `x_i` is assigned to cluster **5**, this means that `c(i)` is 5 and so `mu_c(i)` is `mu_5`.
			- In other words, it means that the centroid of the cluster to which `x_i` has been assigned is the mean vector of cluster 5.
- The cost function is a function of all the cluster centroid indexes to which each sample has been assigned and the mean position of each cluster.
- The optimisation objective is to minimise the average squared distance between the cluster centroid for each cluster `c_i` and the samples assigned to that cluster `x_i`.
- Thi cost function is called the **distortion** of the K-Means algorithm.
- The parameters that are modified for this objective are all the cluster indexes used for cost function and the mean locations of all cluster centroids.
- The **cluster assignment** step is minimizing `J` - the cost function - by modifying cluster indexes `c1, c2, c3, ..., cm` while holding the centroids of clusters `mu_1, mu_2, ..., mu_K` constant.
- The **centroid movement** step is minimizing `J` by modifying the centroid positions for all clusters `mu_1, mu_2, ..., mu_K` while **holding the centroid indices constant**.
- Can used this cost function to make sure K-means is converging and implemented properly.

## Random Initialization
- The first step in the K-Means algorithm is to randomly initialize the cluster centers.
- One method that works better than most other options is as follows.
	- Should have the number of cluster centroids `K` < the number of examples `m`.
	- Randomly pick `K` training examples and set `mu_1, mu_2, mu_3,..., mu_K` to these `K` training examples.
- This means K-Means is a non-deterministic algorithm: it can arrive at different solutions based on how the initial clusters are assigned.
- So K-means may converge to local optima instead of global optima of the distortion function `J`.

### Multiple Random Initializations
- Initialize and run K-means lots of times to find the best initialization of the K-means.
- For `i` = 1 to 100 { # typical range is 50 - 1000 depending on size of dataset, compute resources available
	Randomly initialize K-means.
	Run K-means.
	Get `c(1), c(2), c(3),..., c(i), mu_1, mu_2, mu_3, ..., mu_K`.
	Compute the cost function (distortion)
		`J(c(1), c(2), c(3),...., c(i), mu_1, mu_2, mu_3,..., mu_K`).
}
Pick the clustering that gave lowest cost/distortion `J(c(1), c(2), c(3),..., c(i), mu_1, mu_2, mu_3, ..., mu_K)`.
- Useful if K = 2 to 10 but if K is very large (K >> 10 e.g. 100s) - diminishing returns.

## Choosing the Number of Clusters
- Number of clusters = the value of the parameter `K`.
- No straightforward/analytical solution
- The most common way is to do so manually
	- Visualizations
	- Looking at outputs etc.
- Genuinely ambiguous how many clusters there are in the data: this is part of why unsupervised learning is more challenging than supervised learning.

### Elbow Method
- Vary the total number of clusters `k` from 1 to 8.
- Compute the cost function or distortion `J` with or without random initialization.
- Plot the cost at each number of clusters on a graph that shows how the cost decreases as number of clusters increases.
- There is a clear **elbow** or kink in the graph: number of clusters where the distortion switches from going down rapidly to going down very slowly.
- After the elbow, there are fast diminishing returns in cost optimisation with increasing number of clusters.
- Not used very often because the elbow very often isn't apparent or obvious for many unsupervised learning problems.

### Downstream Purpose: What are you using `K` for?
- Sometimes you're running K-means to get clusters to use for some downstream purpose. 
- Evaluate K-means based on a metric that captures how well a specific number of clusters performs for that later purpose. 
- E.g. in our t-shirt example, we could choose `K = 3` for three sizes: small, medium, large, or we could choose `'K = 5` for five sizes: XS, S, M, L, XL.
- Use sales/customer satisfaction survey results/complaints/exchanges as a performance indicator for if the S/M/L or XS/S/M/L/XL clusters work better.
