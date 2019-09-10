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

# Advanced Topics