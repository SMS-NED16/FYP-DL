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

# Advanced Topics