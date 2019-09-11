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

# Recommender Systems and Collaborative Filtering