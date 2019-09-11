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
s
# Recommender Systems and Collaborative Filtering