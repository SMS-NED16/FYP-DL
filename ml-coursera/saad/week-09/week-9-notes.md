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

# Recommender Systems and Collaborative Filtering