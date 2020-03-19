# Hands-on Machine Learning - Chapter 8 - Dimensionality Reduction - Exercises

## Question 1 - Motivations for Dimensionality Reduction

### What are the main motivations for reducation a dataset's dimensionality? What are the main drawbacks?
#### Motivations
- **Time Complexity**: Primary underlying assumption is that data in high-dimensional space is not uniformly distributed. It changes very little along some axes, and is highly correlated along others. It makes little sense to retain all the input data for training when the same - or better - performance can be extracted using a subset of this higher dimensional feature space. This will often improve training time for a lot of machine learning models, whose time complexity often depends on the dimensionality of the feature space.
- **Space Complexity**: For the reasons discussed above, it is also beneficial to use dimensionality reduction to compress data so that it occupies less space on a device. 
- **Visualization**: Dimensionality reduction can help condense input feature spaces to 2 or 3 dimensions, which allows them to be plotted on graphs. Visual inspection of these graphs can often grant insights into the relationship or structure of the data which, in turn, can help inform feature engineering and model selection. 
- **Improve Model Performance**: Sometimes, not all input features are actually correlated with the target class. When in abundance, they can act as noise that prevents the model from learning real, ground-truth associations between I/O. Thus, in very few cases, dimensionality reduction can actually improve model performance (in terms of regression/classification error) by eliminating noise. 
- **Simplifying Machine Learning Task**: Also, the implicit assumption for dimensionality reduction is that the machine learning task becomes 'simpler' in lower dimensional space. However, this isn't always the case, and depends on the dataset.

#### Drawbacks
- **Loss of Information**: Dimensionality reduction is essentially compression. Like most compression techniques, the process isn't lossless: projection-based PCA, for instance, will often lose some data in terms of explained variance when transforming high-D data to low-D data. So original data lost through dimensionality reduction.
- **Pipeline Complexity**: Dimensionality reduction can also make machine learning pipelines more complex. 
- **Computationally Intensive**
- **Interpretability**: Transformed features can often be difficult to interpret.

## Question 2 - Curse of Dimensionality

### What is the curse of dimensionality? 
- Most real-world data is often very high dimensional.
- Data behaves very differently in high dimensions: the greater the dimensionality, the greater the chances that a given machine learning model will overfit the data.
- Number of training samples required to achieve uniform distribution of data points in higher dimensional space (so as to avoid overfitting) increases exponentially with feature space dimensionality. It simply isn't possible to find enough training instances to achieve the 'density' of samples required to avoid sparsity in higher-dimensional feature space.
- Sparsity is bad because a new test set example is likely to lie 'very far away' in the high-dimensional space from other training set examples on which models have been trained.
- This means that models trained on higher-dimensional data are usually not as robust as those trained on lower-dimensional data, and likely to overfit. 
- In high dimensional space, the likelihood that any given set of features is 'extreme' across any given axis is almost guaranteed, which means these samples are likely to skew the actual machine learning model trained with the data.

## Question 3 - Reversing Dimensionality Reduction

### Once a dataset's dimensionality has been reduced, is it possible to reverse the operation. If so, how? If not, why not?

It really depends on the dataset and the method being used for dimensionality reduction.

#### Possible when...
- Using singular value decomposition (as in PCA) and have access to all three constituent matrices $U, \Sigma, V_T$ returned by `svd` or a matrix of all principal components. In this case, original data can be recovered as a dot product: $dot(U, \Sigma, V_T)$, which is exactly what happens when we use `PCA`'s `inverse_transform`. 

#### Partially possible when
- the cumulative explained variance ratio of the principal components $\epsilon$ is $< 1.0$ i.e. not all principal components are available. If $\epsilon$ is high, then dimensionality reduction is still somewhat possible. 
- the dimensionality reduction uses a highly non-linear projection kernel such as an `RBF` kernel that maps the input space to an infinite-dimensional feature space before reducing this to a lower-dimensional feature space. Partial reconstruction is possible by finding the preimage of the points corresponding to the reduced dimensional space in the original `n`-dimensional feature space through a kernel trick mapping between the high dimensional feature space and `n`-dimensional space. 


#### Not possible when
- all principal component vectors are not available.
- kernel function for preimage is not available.

## Question 4 - PCA and Non-Linear Datasets

### Can PCA be used to reduce the dimensionality of a highly non-linear dataset?
- Yes. 
- Use a kernelized PCA instead of vanilla (linear) PCA.
- Uses a kernel trick to map `n`-dimensional space to very high dimensional feature space (possibly $\infty$-dimensional feature space) where the data is linearly projectable onto a `d`-dimensional, low-D reduced space.
- Kernel functions can transform non-linear datasets into linear datasets, and thus make linear projections possible. 

## Question 5 - Dimensions After PCA

### Suppose you perform PCA on a 1000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?

- It really depends on the dataset. Can't say for sure how many dimensions the reduced feature space will have. 
- 95% variance could be preserved by as few as 1 feature in the 1000-dimensional space, or by as many as 999. Again, every dataset is different.
- What I can say for sure is that if 95% - rather than 100% - of the explained variance in the original dataset is accounted for, then `d` will always be lower than `n`.
- $d < 1000$.

## Question 6 - Use Cases

### In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?

#### Vanilla PCA
- Linear (or near-linear) dataset. 
- Entire dataset can be fit into memory at once.

#### Incremental PCA
- Linear (or near-linear) dataset.
- Entire dataset cannot be fit into memory at once.
- Slower than vanilla PCA.
- Online learning. 

#### Randomized PCA
- When the dimensionality of the reduced space `d` is less than 80% either the dimensionality of original space `n` or the number of training examples `m`.
- Considerable reduction in dimensionality required.
- Data fits in memory.

#### Kernel PCA
- Highly non-linear dataset. 

## Question 7 - Evaluating Performance

### How can you evalaute the performance of a dimensionality reduction algorithm on your dataset?
- Dimensionality reduction works well if it eliminates a lot of dimensions from the dataset without losing too much information.
- One way of gauging this is to measure the reconstruction error between the original data and the data obtained by applying a reverse transformation to the reduced features. 
- If the dimensionality reduction is being done as a preprocessing step for a machine learning algorithm, the machine learning algorithm's loss (or other performance metric) can be used to gauge how good the dimensionality reduction algorithm was.
    - If the right dimensionality reduction algorithm was used, not a lot of information was lost.
    - If not a lot of information was lost, the algorithm should perform just as well when using the original dataset.

    ## Question 8 - Chaining Dimensionality Reduction

### Does it make any sense to chain two different dimensionality reduction algorithms?
- Yes, it makes sense to chain two different dimensionality reduction algorithms. 
- Different dimensionality reduction algorithms operate at different levels of granularity: some are very good at getting rid of lots of useless dimensions (e.g. PCA) while others are much better at local transformations (LLE).
- Common approach: apply PCA, then LLE. PCA quickly reduces a lot of dimensions, LLE takes less time to eliminate further dimensions, even though it is usually slower. 