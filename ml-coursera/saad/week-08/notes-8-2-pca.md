# Coursera - Machine Learning - Week 8

# 8.2 - Dimensionality Reduction

## Motivation 1 - Data Compression
### What is Dimensionality Reduction?
- Another unsupervised learning technique.
- Suppose we've collected a dataset with many features and the chosen to plot 2 of them.
- Unknown to us, `x1` is length in meters while `x2` is length in feet - highly redundant.
- Want to reduce this 2D data into 1D data since having an additional feature gives no additional information.
- Not an unrealistic problem:
	- Different engineering teams give different features, usually 100s or 1000s.
	- So difficult to keep track of which features are unique.

### Example - Helicopter Pilots
- Another example: helicopter pilots
	- x1: pilot skill
	- x2: how much the pilor enjoys flying
	- What we're really interested is pilot aptitude: a single feature that depends on both the skill and enjoyment.
- Basically projecting features from a higher dimensional space to a lower dimensional space.
- This means we can use only one (or `n - 1`) real numbers to represent the same information.
- TLDR: projecting the redundant features (x1, x2) on a single line that establishes the relatonship between them, and then using only a single real number to represent the information.
- Saves memory because we need only 1 number instead of 2 to store our data/information.
- Also allows us to make algorithm work fasters. 

### Example - 3D Point Cloud
- Can also reduce data from 3D to 2D, or, more realistically, 1000-dim data to 100-dim data.
- E.g. for 3D -> 2D, project the data onto a 2D plane (not necessarily the x-y plane).
- Now we need only two numbers to represent each example - data can be stored with less memory.
- The new feature is called `z_i`, or `z_j`.

## Motivation 2 - Visualization
- For a lot of ML applications it really helps us to visualize data to help understand it better.
- Suppose we have a large dataset of many features about countries around the world	
	- x1 - GDP - x2 - per capita GDP - x3 - HDI
	- x4 - Life expectancy - x5 - Poverty index - x6 - Mean household income
- Realistically, such a dataset would have as many as 50 features - how to visualize? Can't plot 50-D data.
- Using dimensionality reduction, instead of having each country represented by a 50-dimensional feature vector `x_i`, we can come up with a different set of feature transformations `z_1`, and `z_2` to try and understand the data.
- Reduce the 50-dim data to 2-dim data and plot it. 
- Dim reduction does not ascribe a meaning to the new features - this is for us to figure out. E.g. `z_1` could correspond to the overall economic size of the country, whereas `z_2` represents the per-person economic well-being.

## PCA - Principal Components Analysis
- The most commonly used algo for dimensionality reduction.

### Problem Formulation
- Assume we have a 2-dimensional dataset `R2` and we want to reduce it to 1-D data.
- We want to find a line onto which we can project the data in our scatterplot.
- A good line for projection: distance between original point and projected point on the line is very small.
- PCA finds a projection surface that minimises the **projection error** between the original data points in `R_n` and the lower-dimensional space.
- Good practice to perform mean normalization and feature scaling **before** projection. 
- The larger the distance between the original point and the projection, the higher the projection error.
- More formally, find a vector `u_1` in `R_n` onto which to project the data so as to minimize the projection error.

### General Algorithm
- Reduce `n` dimensional data to `k` dimensional data.
- Find `k` vectors `u_1, u_2, u_3,..., u_k` onto which to project the data so as to minimize the projection error.
- E.g. for 3D -> 2D, project data onto linear subspace spanned by the vectors `u_1, u_2, u_3, ..., u_k`.

### How does PCA Relate to LinReg?
- PCA is **not** linear regression, and all similarities are cosmetic.
- For linear regression, we'd be trying to predict the value of `y` using some features `x` by fitting a straight line that is formed by minimizing the mean squared error between predicted and actual values.
- In PCA, the projection errors are **orthogonal distances** or shortest distances between the data points and the lower-dimensional surface.
- Also there is no special variable `y` that we are trying to predict. We simply trying to map higher dimensional data to lower dimensional data.

## PCA Algorithm Implementation
### Data Preprocessing
- Suppose you have a training set `x_1, x_2, x_3,..., x_m`.
- Perform **mean normalization** and **feature scaling**
	- Mean `mu_j` = 1/m * sum(i, m)[x_i_j] 
		- This is the mean of the `j`th feature, found by adding the `j`th feature's value for the `i`th training example where `i` = [1, 2, 3,..., `m`]
	- Replace each `x_i_j` with `x_i_j - mu_j`.
	- If different features on different scales, scale features to have comparable range of values.
		- `s_j` = max_j - min_j
	- So `x_i_j` = (`x_i_j` - `mu_j`) / `s_j`.

### Dimensionality Reduction
- Recall that PCA tries to find a lower dimensional subspace on which to map training examples to minimise the projection error.
- Find a set of `k` vectors `u_1, u_2, u_3, ..., u_k` which define a lower dimension subspace.
	- E.g. if `x` is 2-dimensiona, map it to 1-dimensional space spanned by `z_i` which is in `R_1`.
		- Straight line along one axis. Just map points along the line.
	- E.g. if `x` is 3D, map it to 2D space spanned by `z_1_i, z_2_i` which is in `R_2`
		- 2D plane in 3D space.
- How to compute `u_1, u_2, ..., z_1, z_2,...`? V complicate! Doin me a heckin bamboozle. 

### Algorithm
Reduce data from `n` dimensions to `k` dimensions `n > k`.
Compute the covariance matrix
	`Sigma` = 1/`m` sum(`i` = 1, `n`)((`x(i)`)(`x(i))Transpose`)
	Vectorized implementation in Octave/MATLAB is `1/m * X' * X'`.
Compute the eigenvectors of the covariance matrix
	`[U, S, V] = svd(sigma)`
	- Could use `eig` but `svd` i.e. single value decomposition is more stable.
	- `svd` and `eig` are different functions, but when applied to covariance matrices, that are **symmetric positive semi-definite**, they result in the same thing.
	- `Sigma` is an `(n, n)` matrix
		- `x(i)` is `(n, 1)` vecot
		- So `x(i)_transpose` is an `(1, n)` vector
		- So their product is `(n, n)`.
- `U` will be the `n x n` eigenvector or basis matrix.
- To consider the `k` dimensional subspace, we use the first `k` columns of the `U` matrix. 
- To map `x` (which is an `n` dim vector) to `k` dimensions
	- Consider the basis for the low-dim space i.e. `U_reduced`.
	- Compute transformation from `n` to `k` dim space using `U_reduced_transpose` to X.
	- `U_reduced_transpose` will have dimensions `(k, n)` and the X will be `(n, 1)` so the resulting vector will be `(k, 1)`.

## Reconstruction from Compressed Representation
- How to go back from `z_i` (say 100 dimensional) to `x_i` (say 1000 dimensional)?
- Recall that `z` = `U_reduce_transpose` * `x`.
- Given the point `z` in `R_k` can we map it back to `x` in `R_n`?
- `x_approx` = `U_reduce` * `z`
	- If projection error is not too great, the reconstructed points will be close enough to their original values.
- Given an unlabeled data, we now how to map the data between high-dim and low-dim as well as the reverse.

## Choosing `K` - Number of Principal Components
- A parameter of the PCA process called the **number of principal components**.
- What PCA tries to do is minimise the avg squared projection error
	`1/m * sum(i, m) * square(x_i - x_i_approx)`
- Total variation in the data
	`1/m * sum(i, m) * square(x_i)`
	On average, how far are my training examples from just being all zeroes i.e. distance from origin?
- Choose a value of `k` such that 
	`avg_squared_projection_error` / `total variation` <= 0.01 i.e. 1%
- 99% of the variance in the data is retained.
- Other common threshold values: 0.05 (5% of variance retained), 0.10 (10% retained), and as low as 0.15 (85% of variance retained)
- Most common is 95% - 99%.
- Surprising how for many datasets we many features we can reduce while still retaining 99% variance - lots of features are correlated so we can easily drop most features without compromsing predictive power. 

### Algorithm 1
- Try PCA with `k` = 1
- Compute `U_reduce`, `z_1, z_2, ..., z_m`, `x_1_approx, x_2_approx, x_3_approx,...,x_m_approx`.
- Check if avg squared error / variance <= 0.01
- Keep incrementing `k` as long as the condition is satisfied. 

### Algorithm 2
- Using `[U, S, V] = svd(sigma)` gives us `S`, an `n, n` square diagonal matrix (all non-diagonals are zero).
- Find the diagonal sum of the `S` matrix for the first `k` and `m` diagonal elements - `S_k` and `S_m`.
- If `(1 - S_k) / S_m` is <= 0.01, keep increasing `k`.
- Don't need to call `SVD` again and again to test different values of `k`, or run PCA from scratch over and over again.
- Alternatively, could just compare `S_k / S_m` with the required degree of variance to maintain i.e 99, 95 or 85%.

## Advice for Applying PCA
- How to improve running times of learning algorithms using PCA in practice.

### Supervised Learning Speedup
- Suppose we have a supervised learning problem (features `X` and labels `y`) and features are very high dimensional `n = 10k` e.g. in computer vision (10k pixel intensity values).
- Extract inputs `X` without labels to form an unlabeled training dataset `x_1, x_2, x_3,...,x_m` (each is 10k dimensional vector).
- Now apply PCA to transform `X` to a 1000-dimensional vectors `z_1, z_2, z_3,...,z_m`.
- Zip these with the original labels/targets and feed it to a new learning algorithm. 
	- E.g. using logistic regression, `h_theta(z)` = 1 / (1 + e^(-`theta_transpose` * `z`))
- For test examples, use the same `X` to `Z` mapping and predict.
- `[U, S, D] = svd(sigma)` helps **learn** a lower dimensional mapping from a higher dimensional one. It should only be learned using the **Training Data**, and can then be applied to test and CV sets.

### TLDR: Traing/Test/CV
- When running PCA, run it only on the training set and not the CV and test sets. 
- This defines the mapping from `X` to `z` which can then be used to map test and cv training examples. 

### Practical Advice
- Can actually achieve a 10x or 5x decrease in the number of features.

### Summary - PCA
- Benefits of PCA
- Compression
	- Reduce memory/disk needed to store data
	- Speed up the learning data
	- Often without compromising predicting power/variance (~99% retained).
- Visualization
	- Use `k` = 2 or 3 so that we can plot them with MATLAB/Python packages. 

### Misuse of PCA: Preventing Overfitting
- The reasoning
	- The more featues we have for a limited amount of data, the more likely we are to overfit.
	- If we use `z_i` instead of `x_i`  then we are less likely to overfit.
- This is bad application of PCA.
- It might work, but not as well as regularization.
- PCA does not use labels `y`. It just looks at input features `X`, and so it throws away some dimension of your training data without knowing how the output is affected, so might throw out some useful data. 
- Regularization knows what the values of `y` are so it is unlikely to throw away dimensions/data that would reduce predicting power of the training set.

### Example - Misuse of PCA
- Proposed Design of ML System
	- Get training set `{(x1, y1), (x2, y2), (x3, y3), ..., (xm, ym)}`
	- Run PCA to reduce `x_i` in dimension to get `z_i`.
	- Train logistic regression on `{(z1, y1), (z2, y2), (z3, y3),...,(zm, ym)}`.
	- Test on the test set: Map `x_test_i` to `z_test_i` and run `h_theta(z)` on the `{(z1, y1), (z2, y2), (z3, y3),...,(zm, ym)}`.
- Before implementing PCA, but try solving your ML problem using only training set without reducing its dimension.
- Use PCA only if this does not give satisfactory results