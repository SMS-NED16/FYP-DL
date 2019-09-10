# Coursera - Machine Learning - Week 10 Notes

# Gradient Descent with Large Datasets
- One of the reason why ML algos work so well now compared to 5 - 10 years ago is because of the increase in the amount of available data.
- We'll talk about the algorithms that help us efficiently process these large datasets.

## Why?
- High performance: low bias ML algorithm trained on a large dataset.
- It is not who has the best algorithm, it's who has the best data.
- Learning with large datasets comes with its own unique computational problems. 
- Normal size: 100M samples.
- E.g. for gradient descent in linear regression, we need to carry out a summation over a 100M terms to compute a derivative and compute a single step of gradient descent.
- Computationally very expensive. 
- Techniques exist to replace this algorithm or find more efficient ways to use the same algorithm.

## Which models will benefit from large datasets?
- Why not use a random subset of 1000 examples and train the algorithm on it?
- Before investing the effort into developing software to train on millions of samples, its a good idea to check if a few samples will do as well.
- If we are facing a supervised learning problem with a very large dataset, we can confirm if the data is likely to perform better than when using a small dataset.
	- This is done by **plotting a learning curve for a range of values of `m` and verifying that the algorithm has a high variance when `m` is small**.

### Learning Curves
- Training objective function error increases with training set size and cross validation set objective function error decreases with training set size. A large difference exists between the plateuaus of the two training curves.
- This is characteristic of a high variance learning algorithm - having more training examples will improve performance.
- But if the training objective and cross validation objective function will converge roughly to the same error, it is a **high bias** learning algorithm, and will not benefit from additional training examples.
- However, we can add extra features (or hidden units in case of neural network) to a high bias algorithm to improve its variance, and in turn improve the potential for improvement with increased sample size.

## Stochastic Gradient Descent
- Many learning algorithms were based on deriving a cost function or optimization function which is minimised using another algorithm like gradient descent.
- But ordinary gradient descent becomes very inefficient for large datasets.
- Recall that the when minimising the cost function of gradient descent, the parameters of the learning algorithm (as a demo, consider linear regression) move along an `n`-dimensional space to their (hopefully) global minimum. 
- Computing the derivative term requires summing over all `m` examples - can be very expensive. 
- E.g. 300M people in the US census data, and the data itself will have several features.
- This conventional method is called **batch gradient descent** because we're looking at the entire batch of training examples at a time.
	- Will need to stream through the entire 300M records into memory in chunks - can't load into memory directly. 
	- For a single step of gradient descent!
	- And then repeat until convergence.
- Stochastic gradient descent doesn't look at all training examples in the training set - only a **Single training example**.
- The cost function for SGD represents how well the parameters of the hypothesis function are doing on a **single** training example `(x^i, y^i)`.
- The overall cost function `J_train` is just the average cost over `m` training examples.
- SGD scans through the training examples and looking at only the first example, will modify the parameters a little bit to fit just the first training example a little bit better. 
- Then move on to the second training example and move the parameters to fit the second training example a little bit better.
- Repeat until parsed the entire training set. 
- This is why we randomly shuffle the dataset. 
- Don't wait for the cost function to parse all training examples in the dataset. Use just one training example at a time to improve the parameters.
- BGD will take a relatively straight line to to the global minimum, but since SGD looks at individual data points, the direction of descent will be a lot more random - won't be a straight/continuous path to the global minimum. 
- **Generally, but not always** will make its way to the global minimum.
- Does not converge in the same sense as a BGD, and keeps wandering around the global minimum - not a problem in practice because so long as the parameters are in some region close to the global minimum.

**TLDR: SGD - one step per each training example; BGD - one step per epoch (entire training set)**

### Some key points
- When the training set size `m` is very large, SGD can be much faster than BGD.
- The cost function should go down with eveyr iteration of BGD (assuming well-tuned learning rate alpha), but not necessarily with the SGD algo.
- Before beginning the main loop of SGD, it is a good idea to shuffle the training data into a random order.

## Minibatch Gradient Descent
- Another variant of the SGD and BGD algo that can sometimes work faster than SGD.
- BGD: Use all `m` training examples in each iteration.
- SGD: Use a single example in each iteration.
- MBGD: Use `b` examples in each iteration, where `b` is the **mini-batch size**.
	- Typical choice is 10 (range is 2 - 100)
- Get `b` = 10 examples from the training set.
- Compared to BGD, still allows us to make progress much faster
	- Weights/parameters of the model are updated after the first 10 examples instead of all `m` examples.
- But also solves some probles inherent in SGD.
	- Why do we want to look at a batch of samples rather than an individual sample?
	- If we have a good vectorized implementation, we can partially parallelize the gradient computations over the examples.
	- If we just did single examples at a time, we would have less to parallelize over.
- Disadvantage: may need to fiddle with the mini-batch size `b`.
- MBGD becomes the same as batch GD if `b = m`.

## SGD Convergence
### Checking for Convergence
- For small BGD problems, we would plot the optimisation cost function against the number of iterations.
	- When the cost function's value plateaued or reached a constant value, it had converged. 
- But this isn't practical for large datasets - requires sum over the entire data set.
- For SGD, compute the cost as the MSE between the predicted and actual value for a single training example.
- Then during learning update \theta using that training example **after** computing cost. 
- Repeat this for all trainign examples until convergence.
- As SGD is scanning through the dataset, just before updating theta, find the MSE on that specific training example. 
	- If did this after updating theta, it might not be representative of the actual MSE because we just tuned theta to fit that training example.
- After every 1000 iterations plot cost(theta(x^i, y^i)) averaged over **the last 1000 examples**.
	- This gives a running estimate on how well the algorithm is doing on the last 1000 examples.
	- Doesn't cost much to compute the cost and/or averaging the last 1000 costs compared to computing the cost across all `m` examples.
- Because computed over 1000 samples, the averages will be noisy: may not consistently decrease. 
- With a smaller learning rate, the algorithm will take longer to converge but will converge at a smaller error.
	- This is because the parameters will oscillate around the global minimum in SGD.
	- Smaller learning rate means smaller oscillations about the minimum.
	- Usually, smaller learning rate means smaller errors.
- If we **increase the number of training examples** over which the average error is computed, we will get a **smoother curve**.
	- But feedback on the algorithm's performance will be "delayed" - aggregated after 5 times as many samples.
- If the cost is not decreasing at all, it may be possible that the algorithm just isn't learning.
	- Could increase the number of samples over which the error is averaged, as this helps minimise noise and highlight the overall trend - which could be a decrease.
- If a curve has an increasing error with increasing iterations, the algorithm is **diverging**.
	- Use a smaller learning rate alpha.

### Learning Rate
- With SGD, the algorithm won't exactly converge, but will oscillate about the global minimum.
- **The learning rate for SGD is usually held constant**.
- To allow for convergence, we can **slowly decrease the learning rate \alpha over time.**
- Learning rate `alpha = const1 / (iterationNumber + const2)`
	- `iterationNumber` is the number of SGD iterations run (number of training examples seen)
	- `const1`, `const2`, are additional algorithm parameters that have to be tweaked.
- Parameter tuning makes it difficult for alpha reduction to work well, so most people don't use it.
- But if it done properly, the learning rate will decrease as the parameters approach the global minimum, and so the algorithm takes smaller and smaller steps.

### Key Points
- If we reduce the learning rate alpha and run SGD long enough, it is possible that we may find a set of better parameters than with larger alpha.
	- The smaller learning rate can mean the variations about the global minima are smaller, so that the error between the actual analytical solution and numerical solution is smaller, so the cost will be lower and the parameters can be considered `better`.
- If we plot `cost(theta(x^i. y^i))` (Averaged over the last 1000 examples) and SGD does not seem to be reducing the cost, one possible problem may be that the learning rate alpha is poorly tuned.
	- Poorly tuned could mean that it is large enough to cause noise or variations about the actual cost which hides the overall trend.
	- Poorly tuned could also mean that the learning rate is **too** large, and is causing the cost to diverge rather than converge.

# Advanced Topics
## Online Learning
- A new large-scale ML setting that allows us to model problems in which we have a continuous stream of data coming in and would like the ML algo to learn from it..
- FAANG: continuous stream of data created by a continuous stream of users helps algos learn user preferences.

### Example - Online Shipping Service
- Suppose you are running a shipping service (packages from loc A to loc B)
- Users tell you pkg origin and destinations.
- Based on shipping price, users tend to use the shipping service (positive example) and in other cases they don't (negative example).
- Features `x` capture properties of the user (origin, destination, price).
- Want to learn what is the probability that they will elect to ship the package given a specific set of features. `p(y = 1 | x)`
	- This can help us pick a price that will maximise turnover and also maintain profit margin.
- Will use logistic regression for this problem.
- The algorithm for this problem is 
` repeat forever {								// as long as you keep receiving a stream of users
	Get (x, y) pair corresponding to the user 	// features and whether the user chose to use shipping service
	Update theta using (x, y) only				// using the current training example - then discard: never reuses
		theta_j = theta_j - learning_rate * (h_theta(x) - y) * x_j for all j 
}
`
- If data is essentially unlimited (large userbase), then no need to reuse a single sample for training the model.

### Effect of Online Learning Algos
- Can adapt to changing user preferences.
	- Maybe users start to become more or less price-sensitive.
	- Maybe users have different preferences or locations.
- The online learning algo will be able to keep track of this.
- This is because as the pool of users changes, the updates to the parameters also changes.

### Example - Product Search
- Apply learning algo to give good search listings to user.
- Run an online store that displays `n` of all `m` phones in response to a user search query. 
- For each phone given a specific query, we can construct a feature vector `x` and we want to learn `p(y = 1 | x, theta)` so that we can show a user phones they are likely to buy.
- `y = 1` if the user clicks on the phone. 
- This problem is called the problem of learning the **predicted clickthrough rate** - predicted CTR.
- Will run ten different gradient descents based on the 10 phones shown to a user in response to a query. 

### Other Examples
- What special offers to show the user?
- Which news articles to show the user on a news aggregator website?
- Collaborative filtering.

### Key Points
The advantages of using an online learning algorithm are
1. It can adapt to changing user tastes (i.e. if `P(y | x, 0) changes over time`)
2. It allows us to learn from a continuous stream of data since we use each example once and then no longer need to process it again.