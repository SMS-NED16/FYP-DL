# Machine Learning Crash Course - Notes 01: ML Concepts

## Introduction 
Practically, ML helps us do three things
1. Reduce time programming
	- Don't have to hardcode rules.
	- Feed examples to an off-the-shelf ML tool. 
2. Customize and Scale Products
	- Produce an english spelling collector by writing code by hand.
	- But can't scale this to hundreds of other languages.
	- But with ML, moving to other languages is just as easy as collecting data and feeding it into the model.
3. Complete "unprogrammable" tasks
	- Facial recognition, object detection, perception, etc.
	- Don't need to explicitly program the algorithm.
	- Let it learn the rules for solving a problem by exposing it to a lot of data.

ML Changes the way we think about problems.
- Shifts from mathematical focus to natural science.
- Using experiments, statistics, and uncertainty to analyse the results of an experiment.
- Think like a scientist, not like a programmer.

## Framing
How to frame a task as an ML problem, or to decide if this is even possible.

### Basic Terminology
**Supervised ML**: systems learn how to combine inout to produce useful predictions on previously unseen data.
- When training, provide the model with **labels**: e.g. spam/no-spam (the target we are trying to predict).
- **Features** are input variables describing our data. `x1, x2, x3,..., xn` where `n` is the number of features. 
- **Example** is a particular instance of data **X**.
	- Can be **labeled** i.e. has features as well as label `y` 
	- Or **unlabeled**   i.e. has features but no label `y`.
- **Model** is what does the predicting. 
	- It is what we will try to create to predict labels `y'`
	- Defined by internal parameters, which are learned by exposure to data.
	- A relationship between features and labels.
	- Training: showing the model labeled exaples and enabling the model to gradually learn the relationships. 
	- Inference: Applying trained model to unlabeled examples.
- A good feature is a quantifiable signal that likely has a strong impact on the value of a label.
- A useful label is an observable, quantifiable metric that is associated with a set of features.

## Descending into ML
### Linear Regression
- Scatterplot of housing price against housing square footage. 
- Fit a straight line that seems to "fit" the dataset - roughly equal distribution of points above and below the line.
- This is the regression line: `y` = `w1`x1 + `b`
	- `w` because weight
	- subscript `1` because we may later work with higher dimensional data.
	- `b` is the bias i.e. the y intercept.
- Define loss as the quantifiable difference between the predicted and actual price for a house of a specific size `y - y'` as a measure of how good our regression line is as a predictor.

### L2 Loss 
- aka squared error
- Square of the difference between prediction and true value
- (observation - predicton)^2
- The sum of the squared error is computed over the entire dataset.
- It is then divided by the number of elements in the dataset to get the average L2 loss over all examples.
- L2 loss does not just depend on the number of points the hyperplane passes through, but more on how far these points are on average.

Training a supervised ML model means examining many examples and attempting to find a model that minimizes loss. This is called **empirical risk minimization**.

## Minimizing Loss
- By moving parameters in the direction that causes loss to decrease.
- Hyperparameters - configuration settings used to tune how the model is trained.
- Derivative of (y - y')^2 w.r.t weights and biases tells us how loss changes for a given example.
- Learning rate determines how many steps it takes to minimise the loss.
- For convex problems, there is just one minimum in the loss, so it doesn't matter what we initialise the loss to be.
- But for non-convex problems like NNs, it makes a difference: loss function is shaped more like an egg shape, so many different minima, so initialization matters.
- **Gradient descent**: Compute the gradient (the direction in which to move the weights of the model) over all training examples to get the most accurate estimate of the direction. (**Batch gradient descent**)
- But this is computationally very expensive.
- **Stochastic gradient descent**: choose one random example, then compute the gradient based on its loss.
	- More overall steps, but the total computation is lower.
- **Mini-batch gradient descent**: chooes small batches of training examples, compute their average loss.
	- Intermediate solution.
- Empirically, we have found that batch gradient descent is not necessary, and mini-batch is the optimal solution for the gradient descent problem.


### Playground Exercise 1
- This is very cool.
- With a learning rate of 3, found that the loss function initially optimised theta_1 to have a larger value, and then there was instability in the value of theta_2 after convergence.
- The lower the training rate, the more epochs it takes for the model to converge to a solution (decision boundary) with minimal loss.
- On convergence, both params seem to have roughly equal magnitudes.
- Training loss decreases but test loss increases post overfit-onset.