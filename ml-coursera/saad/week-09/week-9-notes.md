# Coursera Machine Learning - Week 9

# Anomaly Detection
## Anomaly Detection Motivation
- A reasonably commonly used type of unsupervised ML but has some aspects that are similar to supervised learning.

### Example - Aircraft QA
- Manufacturer of aircraft engines: QA testing
- Measure features for each engine that comes off the production line.
	- Heat generation `x1`
	- Vibrations `x2`
- Anomaly detection problem is to find if an aircraft engine with a set of features `x_test` is anomalous i.e. differs too much from the other tested engines.
- If `x_test` lies too far away from the cluster of previously tested points - anomalous, very different.

### Density Estimation
- Given a dataset `x_1, x_2, ..., x_m` that we assume are non-anomalous.
- We want to build a model for `p_x` which is the probability that an aircraft engine with the set of features `x` is **non-anomalous**. 
- We define some threshold `epsilon` and if `p(x_test) < epsilon` then we flag it as an anomaly. 
- Alternatively, if the `p(x_test)` is >= `epsilon`, then we say that the example is not anomalous.

### Applications of Anomaly Detection
- Fraud Detection
	- If we have many users and we are able to track their activities (on website or plant)
	- We can make a set of features for the user's behaviour
		- `x1` - how often does the user log in
		- `x2` - number of pages visited
		- `x3` - number of posts on a forum
		- `x4` - typing sspeed.
	- Based on these features, we can model `p(x)` and try to identify users that are behaving strangely and demand further identification/review of their profiles.
	- Tend to flag users that behave **unusually**, and not just fradulently.
- Manufacturing 
	- Airplance engine QA was a good example of this application.
- Monitoring computers in a data cluster
	- If we have a lot of machines in a data center, we can capture these features
		- Memory use 
		- Number of disk accesses/second
		- CPU load
		- Network traffic
	- Based on these features, we can model the probability `p(x)` that the machine is behaving normally or unusually. 
	- Refer such machines to a system admin for further review.

## Gaussian Distribution
- Assume `x` is a real-valued random variable (random number).
- If the probability distribution of `x` is Gaussian with mean `mu` and variance `sigma^2`, we can write this as `X ~ N(mu, sigma^2)`  where the `~` means distributed as and `N` represents the Gaussian or Normal distribution.
- Parametrized by mean `mu` and variance `sigma^2`.
- The bell shaped curve is found by plotting the `p(X)` for different values of `x` for a fixed value of `mu` and `sigma`. 
- Sigma is the standard deviation and signifies the width of the Gaussian distribution.
- Area under the bell-shaped curve for the Gaussian distribution is always 1 because it represents the probability of the random variable having a specific value.
- If the standard deviation decreases, the distribution gets taller because the range of random values is lower.
- If the standard deviation increases, the distribution becomes wider and lower.
- If the mean increases or decreases, the center point of the bell curve shifts along the x-axis.

### Parameter Estimation Problem
- Suppose we have a dataset with `x1, x2, x3,..., xm`, all of which are real numbers.
- We want to test the hypothesis that each of these variables came from a random distribution.
- Parameter estimation is the problem of, given a dataset, figure out the values of `mu` and `sigma` that best represents the Gaussian distribution for that dataset.
- `mu` is the average value of all training examples.
- `sigma` is the average value of the sum of squares of differences between the training example and mean value.
- The values of `mu` and `sigma` are the maximum likelihood primes of the estimates of the mean and standard deviation.
- The denominator in the computation of variance can be `m` or `m - 1`, and each has different mathematical properties, but because the dataset size `m` in ML is usually so large, this does not make much of a difference.

## Anomaly Detection Algorithm
### Density Estimation
- Assume me have an unlabeled training set of `m` training examples, and each example is a set of `n` real-numbers representing `n` features.
- Our goal is to create a model `p(x)` that computes the probability of a given set of features representing an anomalous or non-anomalous sample (depends on the positive class)
- `p(x)` is the product of probabilities of each feature `x_j` being parametrized as a normal distribution with mean `mu_j` and variance `sigma^2_j` for all `n` features.
- Each feature has its own mean and standard deviation and is represented by its own Gaussian distribution.
- Corresponds to an **independence assumption** between the features, but practically works well even if the features are not independent. 
- This problem is called **density estimation**.

### Algorithm
1. Choose features `x` that you think might be indicative of anomalous examples.
	- Features that would take unusually large or small values for an anomalous example.
2. Fit parameters `mu_1, mu_2,..., mu_n` and `sigma2_1, sigma2_2,...,sigma2_n`.
	- `mu_j` = `1/m * sum(i = 1, m)[x_j^i]`
		- This is just the mean of values of a specific feature for all training examples in the dataset.
	- `sig2_j` = `1/m * sum(i, m)(x_j^i - mu_j)^2`
		- This is just the variance of a specific feature for all training examples in the dataset.
	- Can also come up with a vectorized implementation of this 
3. Given a new example `x`, compute `P(x)`
	- `p(x)` = `product(j = 1, n)p(X_j; mu_j, sig2_j)`

### Example
- We have two different features, each of which is modelled as a normal distribution with its own mean and variance.
- We compute the probability that a set of features represents a non-anomalous sample by multipling `p(x_1; mu_1, sig2_1) * p(x_2; mu_2, sig2_2)`.
- This probability lies somewhere on a surface - the height of the 3D surface represents `p(X)` for a specific combination of `x_1, x_2`.
- We then define a threshold `epsilon` that divides the surface plot into two regions - any probabilities inside the ellipse have probabilities greater than `epsilon` - non-anomalous.
	- Large probability - not an anomaly
	- Small probability - anomaly
- So if a test point lies outside the non-anomalous region, it has a very small probability for `p(X)`.

## Developing an Anomaly Detection Application
### Importance of Real Number Evaluation
- When developing a learning algo (choosing features, etc.) making decisions is much easier if we have a way of numerically evaluating our learnign algorithm.
	- A number or quantified measure of how a specific modification affected the algos performance.
- Assume we have some labeled data of both anomalous and non-anomalous nature (labels are `y = 1`, and `y = 0` respectively).
- Training set: unlabeled training set, but consists primarily of non-anomalous samples. May contain a few anomalous samples, though.
- Cross Validation set and Test Sets: will include examples that are known to be anomalous.

### Train/Test/CV Split
- 10000 good (normal) engines
- 20 flawed engines (anomalous) **y = 1**
	- 20 - 50 is a typical range of values.
- Train/test/CV split is as followes
	- Train: 6k good engines (y = 0)
		- Will be used to fit `p(x)`.
	- CV: 2k good engines (y = 0), 10 anomalous (y = 1)
	- Test: 2k good engines (y = 0), 10 anomalous (y = 1)
- Specific reason for using a 60/20/20 split for train/CV/test.
- Not recommended: 6k in test set, 4k in CV set. Test set not created separately - reuse CV examples for test set is a **bad ML practice**.

### Model Evaluation
- Fit the model `p(x)` on `x_1, x_2, ..., x_m`. Assuming most of these are normal (y = 0).
- On a cross validation/test example, predict
	- y = 1 if `p(x) < epsilon` i.e. is anomalous
	- y = 0 if `p(x) > epsilon` i.e. is not anomalius.
- So anomaly detection algorithm makes predictions for the `y` labels in the CV/test sets.
	- This is somewhat similar to supervised learning.
	- But these labels will be very skewed, because y = 0 will be more common than y = 1.
- Possible evaluation metrics
	- Classification accuracy will not be a good metric because results will tend to be skewed.
	- True positive, false positive, false negative, true negative
	- Precision/recall
	- F1 Score 
- Choosing `epsilon`
	- The threshold that we would use to decide when to flag something as an anomaly.
	- Try different values of epsilon and pick the value of epsilon that does well on the CV set.

## Anomaly Detection and Supervised Learning
- AD becomes very similar to supervised learning: we already know some examples that we know to be anomalous/non-anomalous.
- Then why not just use supervised learning? Why use anomaly detection?

### Anomaly Detection Use Cases
- Very small number of positive examples (y = 1) e.g. 0 - 20 only. 
	- These examples will be saved for the CV and test sets.
	- Not used in the training set.
- But a very large numbr of negative examples (y = 0) examples.
- Mnay different **types** of anomalies. Hard for any algo to learn from positive examples will look like.
- Future anomalies may not look anything like the historical anomalous data.
- Fraud detection: usually small number of fradulent users unless we're a huge online retailer with lots of historically fradulent consumers.
- Manufacturing (e.g. aircraft engines)
- Monitoring machines in a data center

### Supervised Learning Use Cases
- Large number of positive and negative examples.
- Enough positive examples for the algo to get a sense of what positive examples are like.
- Future examples are likely to be similar to ones in the training set. 
- E.g. spam emails can have a large variety of positive examples, but we also have a large **number** of examples with which to learn these varieties. So supervised learning makes sense here.
- Weather prediction (sunny/rainy/etc.)
- Cancer classification

### TLDR
- The key difference is the number of +ive examples: anomaly detection is useful when we can have the model learn entirely from a large number of negative examples.


## Choosing Features
- Anomaly detection performance depends heavily on the features chosen for the algorithm i.e. feature selection/engineering.
- Plot a histogram of the data (a specific feature) before feeding it into an anomaly detection performance.
- If the histogram of the data for a feature looks very skewed, use a **log transformation** of the data usually improves AD performance because it can make the feature more Gaussian.
- Can also replace a feature with `log(x_i + c)` with `c` is an arbitrary constant that we will have to play with.
- Can also use `x^(p)` where `p` is an arbitrary constant.

### Error Analysis for Anomaly Detection
- Similar to the method for supervised learning: plot the error after training the model on all training examples.
- Want
	- p(x) to be large for normal examples x.
	- p(x) to be small for anomalous examples x.
- The most common problem
	- p(x) is comparable (say both large) for normal and anomalous examples.
- The goal is to create a new feature (or set of features) that will help us distinguish anomalous examples fron non-anomalous ones.

### Example - Monitoring Computers in a Data Center
- Choose a feature that take on very large or very small values in the event of an anomaly.
- E.g. if we are monitoring a set of computers in a data center through these features
	- x1: memory use
	- x2: number of disk accesses
	- x3: cpu load
	- x4: network traffic
- We have a hypothesis that CPU load and network traffic grow linearly with each other.
- A common failure case is if one server gets stuck in an infinite loop: network traffic does not grow with CPU load.
- Create a new feature x5 = CPU load / network traffic or (CPU load)^2 / network traffic.
- x5 will take on a very large value if this happens.

## Multivariate Gaussian Distribution
- Can sometimes catch anomalies that univariate GD-based AD cannot.
- Assume we have two features: x1 and x2, each of which are modelled as Gaussian distributions.
- It is possible that for a test example, P(x1) and P(x2) will both have relatively high probabilities, and so the conventional gaussian approach will fail even though in 2D space it lies far away from the nominal region.
- Multivariate approach still uses `n` featues in real space, but instead of modelling them individually, it models them all in one go.
- Sigma in this case is **not** the summation - it is an `n x n` matrix of covariances - usually identity (if we asssume features are independent).
- Can use multivariate distributions to model correlations between data.
	- By changing the off-diagonal values in the covariance matrix. The "direction" of the surface will change.
- Can also vary the values of the mean - position of the peak will change.
- Allows us to capture when two or more features will be positively or negatively correlated. 

## Anomaly Detection with Multivariate Gaussian Distribution
- Formulae for `mu` and `sigma` are same as those used for PCA.
- Original model is a special case of the mutivariate distribution in which the contours are always axis aligned.
- The constraint is that the covariance matrix sigma **must have 0 on the off-diagonal elements**.
	- Because independence assumption.

### When to use each model?
- Original Model
	- Used somewhat more often.
	- Manually create features to capture anomalies where x1, x2 take unusual combinations of values. 
		- E.g. when we want to create a custom feature for anomaly detection. 
	- Computationally cheaper (scales better to very large values of `n` or number of features.)
	- Ok even if `m` is small.
- Multivariate Gaussian Model
	- Used less frequently, but has the advantage of being able to capture correlations between features.
	- Automatically capture correlations between features.
	- Computationally more expensive.
	- `m` must be greater than the number of features `n`, otherwise the covariance matrix is not invertible.
- Rule of thumb: only use multivariate model if `m` is **much bigger** than - greater than 10`n` or ideally `n^2 /2` but the latter is too large.

### When is Covariance Matix singular?
- Non-invertible.
- Failing to satisfy `m` > `n` condition
- Has redundant features
	- Have accidentally created featues that are just linear combinations of other/existing features.
	- The covariance matrix needs to be linearly independent. 
- The odds of this happening are very low. As long as `m` is much greater than `n`, running into a non-invertible covariance matrix is rare.