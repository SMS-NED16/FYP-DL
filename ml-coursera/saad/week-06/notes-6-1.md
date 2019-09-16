# Coursera - Machine Learning - Week 6
# 6.1 - Bias, Variance, and Deciding what to do Next

## Deciding What to Try Next
- There is a difference between people who know when to apply specific algorithms for specific problems.
- Must pursue the most promising avenues for improving performance during ML system design.
- Suppose we are tyring to improve the performance of an ML system e.g. predicting housing prices through regularized linear regression and find that we're making large errors in the predictions.
- What can we do to improve performance?
	- Get more training examples: phone surveys, online forms, door-to-door surveys. 
		- Sometimes getting more data just doesn't help. 
		- So not worth spending months/resources trying to get more training examples.
	- Try a smaller set of features.
		- Carefully selecting a smaller subset to prevent overfitting.
	- Try a larger set of features.
		- Maybe the existing number of features are not informative enough.
	- Adding polynomial features
		- x1, x2, x1 * x2, x2^2
	- Try increasing or decreasing the regularization parameter `lambda` 
- All of these steps require months of preparatin and work, but people tend to make this decision based on instinct/randomness/gut feeling.
- Not a good idea of using time and resources: need to choose the most promising avenues to pursue when improving ML performance instead of pursuing something that does not work.

### Machine Learning Diagnostics
- A test you can run to get insights into what is (or isn't) working in an algorithm.
- How best to improve its performance.
- Diagnostics can take a lot of time to implement and understand, but worth the effort because they can prevent us from pursuing avenues that do not improve ML algo performance. 

## Evaluating a Hypothesis
- When we fit the parameters of a learning algo, we are choosing parameters to minimise the training error.
- But a very low training error doesn't necessarily mean it is a good hypothesis - hypotheses can overfit and fail to generalize to new examples not in the training set.
- How to tell if hypotheses are overfitting?
	- Could plot the hypothesis function's output for low-dimensional data.
- But for higher-dimensional data, have to use **train-test-validation** split.

## Train-Test Split
- Split data into 2 sections (training and test sets) (70-30 split). More data always goes to the training set and less to the test set.
- `m` is the number of training examples, `m_test` is the number of test set examples. 
- If data is not randomly ordered, shuffle or randomly reorder the dataset before sending 70% to training set and 30% to test set.

### Training/Testing for Linear Regression
- Learn the parameters `theta` frm the training data (minimziing the training error `J(theta)`).
- Compute the test set error
	- Linear Regression: `J_test(theta)` = 1/(2 * `m_test`) sum(i, `m_test`)[h_theta(xi_test) - yi_test]^2 
	- Logistic Regression: `J_test(theta)` = 1/(`m_test`) sum(i, `m_test`)[yi_test * log(h_theta(x_i_test))...]
- Can also use the misclassification error for classification problems.
	- Error `e` of the prediction `h(theta(x))` is 1 if 
		- 1 if the hypothesis outputs a value >= 0.5 but the label is 0 
		- 1 the hypothesis outputs a value <= 0.5 but the label is 1.
	 and 0 otherwise.
	- Misclassification error is therefore `1 / m_test` sum(i, `m_test`)[`err(h_theta(x_i), y_i)]`.


## Model Selection - Train/Test/Validation
- Suppose we are trying to choose the degree of polynomial to use for a regression model, or the regularization parameter.
- This is called the **model selection** problem: we're choosing parameters that describe a property of the model - hyperparameter.
- The training set error is not a good predictor for how well the model will generalize to new examples not seen in the training set, especially if the model is being tested on the same data it was trained with.

### Example 1 - Degree of Polynomial for Regression
- Suppose we want to choose a polynomial regression model to a dataset and can choose a degree of polynomial between 1 and 10.
- Can minimise the training error over all models to get different parameter vectors `theta_1, theta_2, theta_3,..., theta_10`.
- Can then compute the test set error for each of these parameters `J(theta_1), J(theta_2), J(theta_3),...,J(theta_10)`.
- Could then use the model with the lowest test-set error e.g. model wth `theta_5`. 
- But using the test set performance is not a good estimate of generalization ability i.e. on data that is not present n the test set: we tuned the hyperparameter - the degree of the polynomial - using the test set.
- So the model will give a better performance on the test set than on **actually unseen data** - it won't generalize as well as the test set results would suggest because hyperparameter tuning creates an implicit **information leak** between the test set and the model. 

### Example 2 - Polynomial Regression with Cross Validation Set
- Instead of splitting the dataset into two sets, split it into three sets: training, cross validation, test.
- Training/cross validation/test have a typical split of 60-20-20. 
- Will still learn parameters using the training data, but will tune hyperparameters (i.e. decide the degree of polynomial) using performance on the cross validation set i.e. `J(theta_cv1), J(theta_cv2), J(theta_cv3),...,J(theta_cv_m_cv)`. 
- We choose the model with the lowest cost `J` on the cross-validation set and use it to make predictions on the test set - this will be a better predictor of generalization performance because the parameter `d` has not been tuned on the basis of the test set

## Bias vs Variance
- If a learning algo does not do as well as we were hoping, we almost have one of the following
	- High bias: underfitting
	- High variance: overfitting
- Plotting the training and cross-validation errors as functions of `d` - the degree of the polynomial.
- Training error
	- As the polynomial degree increases, we can fit the training set better and better, and the training error decreases (possibly to 0).
- Cross Validation Error
	- As the polynomial degree increases, the error is initially very high because we're using a very simple model (straight line) which probably doesn't model the data well.
	- Eventually, as the model's degree increases, the model can fit the data a little better and generalizes well. 
	- However, the error does not keep on decreasing: at one point the error increases because the model has overfit the data and can't generalize to samples in the cross validation set.

### High Bias or High Variance?
- High bias
	- Underfitting
	- Fitting an overly simple model to the data.
	- **Both CV and Training error will be high**.
	- **J_CV approx J_train**
	- CV error may be slightly higher than training error.
- High variance
	- Overfitting
	- Degree of polynomial is too large for the dataset we are trying to model. 
	- **Training error is low but CV error is much larger** 
		- we will be fitting the training set too well!
	- **`1J_CV >> J_train`** 
- By diagnosing whether a model is suffering from high variance or high bias, we can decide how best to improve the performance of the learning algorithm.
- **Optimal model** (in this case, the model with the optimal value of `d`) is the one at which the error changes from decreasing to increasing - inflection point. 

## Regularization and Bias/Variance
- How does regularization interact with/affect bias and variance of learning algos?
- Suppose we're using a regularized 4th order polynomial model. 
	- Large `lambda` - High bias (underfit) since parameters of higher-order terms are heavily penalized. 
	- Intermediate `lambda` - Just right. 
	- Small `lambda` - High variance (overfit). Virtually no regularization, so no penalty for higher order params being larger and overfitting the data.  

### Choosing a Good Regularization Parameter
- Assume the training set cross-validation set errors are simply 1/2 of the average squared error over the training and cross validation sets **without any regularization**.
- Consider a range of values of `lambda`: [0, 0.01, 0.02, 0.04, 0.08,...10] - 12 different values.
- For each value of `lambda`, minimise the cost function `J_theta` to get parameter vectors `theta_1, theta_2, theta_3,..., theta_12`.
- Evaluate the performance of each parameter vector on the cross validation set to identify the value of `lambda` that gives the lowest cross-validation error. 
- Then use this `theta_min` to compute the test set error.

### Bias and Variance
- Large lambda: lower risk of overfitting, higher risk of underfitting - **bias problem**
	- High J_cv and J_train
- Small lambda: higher risk of overfitting, lower risk of underfitting - **variance problem**. 
	- High J_cv but low J_train
- It is some intermediate value of `lambda` which will be **just right** for tradeoff.

## Learning Curves
- A tool used to diagnose if the learning algo is suffering from a bias or variance problem
	- A sanity check: is the algo working correctly?
	- Diagnosing bias/variance problems in algorithms.
- Artificially decrease the training set size: limit yourself to using fewer examples than actually available, and plot the training and cross validation errors. 
	- Training set error is virtually zero for `m` = 1, 2, 3 in the given example. 
	- After `m` = 4/5, the perfect fit is lot: harder and harder to make sure that there exists a single model that fits all points perfectly. 
- What about the cross validation error?
	- Small training set: difficult to generalize well because set is too specific.
	- As data size increases, hypothesis improves, and error decreases.
	- So CV error starts out high and eventually decreasesor plateaus.

### High Bias
- CV error plateaus or flattens out as we increase the number of training examples because it is impossible to fit a better model to the training data of a specific form.
- Performance of CV and Training set will look very similar: high CV and high train errors.
- When a learning algo is suffering from high bias, **getting more data will not help**.

### High Variance
- If training set size is small (and we're assuming `lambda` is also small), we'll definitely overfit the data.
	- This degree of overfitting decreases with increasing training set size.
- Cross validation error will be high and decrease a little bit 
- **Large gap between the training and cross validation error**.
- **Getting more data is likely to help** - think of extrapolating the curve
	- Training error would keep on going up and CV error would keep on going down.

These are idealised curves - IRL they tend to be noisier. But general trend is useful and helps identify bias/variance condition.

## Deciding What to Do Next
- How does evaluation of models/hyperparameter tuning/bias variance tradeoff help us decide what to do next?
 
|Design Decision|Effects|
|--------|---------|
|Get more training examples|Fixes high variance|
|Try smaller sets of features|Fixes high variance|
|Try getting additional features|Not always, but helps high bias problems|
|Adding polynomial features|Fixes high bias|
|Decreasing `lambda`|Fixes high bias|
|Increasing `lambda`|Fixes high variance|

### Neural Networks and Overfitting
- Small neural networks
	- Computationally cheaper
	- Fewer hidden units.
	- Fewer parameters, so more prone to underfitting. 
-  Large neural networks
	- Computationally more expensive.
	- More hidden layers therefore more parameters therefore more prone to overfitting.
	- Use regularization to address overfitting. 
- How many hidden layers to use?
	- A single hidden layer is a reasonable default. 
	- But use a training-test-cross validation split to evaluate NNs with different numbers of hidden layers to evaluate which NN performs best. 