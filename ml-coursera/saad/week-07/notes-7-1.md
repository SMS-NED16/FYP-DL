# Coursera - Machine Learning - Week 7

# Support Vector Machines
- Performance of many supervised learning algos is often very similar, so choice of algo matters less than the amount of data available.
- What matters more is how we use the algo: tuning hyperparameters.
- Despite this, SVM is a very promising supervised learning algorithm
	- important in industry and academia
	- gives a cleaner and sometimes more proper way of learning complex non-linear functions.

## Optimisation Objective
- A modified version of the logistic regression cost function.
- In logistic regression, when y = 1, we want h_theta(x) to be 1, which means theta_transpose * x = z should be much greater than 0.
- Similarly, when y = 0, we want h_theta(x) to be 0, which means z is must less than 0. 
- Modifying the cost function of logistic regression so that 
	- When y = 1, cost should be small ONLY IF z >= 1 -> afterwards cost is 0.
	- When y = 0, cost should be small ONLY IF z <= -1 -> before -1, cost is 0.
- Both are piecewise linear functions called cost_1(z) and cost_0(z).
- In the cost function for SVM, we modify the logistic regression cost function by
	- Replacing the y = 1 and y = 1 logarithmic sums with piecewise linear `cost_1(z)` and `cost_0(z)`.
	- Removing the `1 / m` term by multiplying the entire function with some constant `m`m.
	- The logistic regression function can be considered as `A + lambda * B`
		- A is the sum from the training set
		- B is the regularization term
		- By setting different values of lambda we were trading off between how much we want to optimise the values of the parameters theta to minimise the cost and how much we want to prevent the values of the parameters from growing too large.
	- For SVM, we conventionally use `CA + B` in the cost. 
		- If we set `C` to be a very small value, then the regularization term will have more weight i.e. the parameters will prefer to not become large instead of fitting data.
		- C plays a similar role (**but is not equivalent to**) 1 / lambda.
- Unlike logistic regression, the SVM doesn't output a probability.
- The SVM hypothesis makes a prediction of `y` being equal to 1 or 0 directly.
	- h_theta(x) = 1 if theta_transpose * x is >= 0
	- h_theta(x) = 0 if theta_transpose * x is < 0
	- Not proabilities about the label being 1. Not a continuous valued function.

## Large Margin Intuition
- SVMs are sometimes called **large margin classifiers**.
- Technically, if the hypothesis predicts y > 0, we can assume the actual label is 1. And if the hypothesis is less than 0, we can assume label is 0.
- But support vector machine hypothesis has an additional **margin**
	- Don't just barely get the example.
	- Not enough for the hypothesis output to be just slighly above 0 for the sample label to be considered 1. It must be at least above 1.
	- Not enough for the hypothesis output to be just slightly below 0 for the sample label to be considered 0. It msut be at least below -1.
- Suppose we set `C` to be a very large value e.g. 100k
	- When minimising the objective function, we will be highly motivated to set the training set term `A` equal to 0.
	- This will be subject to the constraints defined above.
- What does this mean for separation?
- Assume that data is linearly separable: there exists a straight line that can separate the positive and negative examples perfectly. 
	- Many such decision boundaries exist, but not all of them are equally good. 
- The support vector is the **optimal** decision boundary. 
	- The distance between the decision boundary and the closest example from each class in the `n` dimensional vector space is the
- For a good decision boundary
	- **If the decision boundary is at 90 degree angle** to most examples in the training set, the lengths of the projections are much bigger.
 (maximum, actually).- Since we still n **margin**.
	- SVM tries to separate the data with as large as a margin as possible.
	- This is a consequare still subject to the same constraints for theta_transpose the optimisation problem that we derived above.
	- The vectors at a distance `margin` away from the decision boundary are called **Support vectors**.

### Large Margins and Outliers
- If `C` is very large, the SVM method is susceptible to being skewed by outliers.
- In practice, when `C` is optimised properly, SVM will ignore outliers **even if the data is not linearly separable**.

## Mathematics for Large Margin Classification
- The missing link: how does the optimisation problem lead to the large margin classifier? 

### Vector Revision
- Dot product between `u` and `v` = `u_transpose` * `v`. This is a real number. 
- Norm of a vector `||u||` is the Euclidean length of the vector `u`. This is always a real number. 
- P is the length or magnitude of the projection of `u` onto `v`. This is also a real number. 
	- P is signed: can  be positive or negative.
	- If angle between `u` and `v` is greater than 90 degrees, then the direction of the projection is opposite to the vector onto which the vector is being projected. 
	- So inner product between two vectors can be negative if the angle between them is greater than 90 degrees. 
- The dot product `u_transpose * v`  is the same as `P` * norm `u` = `P` * `||u||`.
- `u_transpose * v` = `v_transpose * u` so the reverse process will also give the same real number.
- See handwritten notes for more details. 
- Can rewrite the decision boundary optimisation objective in terms of the projection `p` of the features `x_i` onto the parameter vector `theta` and the norm of `theta`.
- For a bad decision boundary, the projection scalar `p_i` will be small for most examples `x_i` 
	- this means we want the norm of theta must be large (because of the constraint y = 1 if theta_transpose * x >= 1).
	- the norm of theta will also be large even if the target class is negative (in this case the projection should be negative).
	- The constraint means that since `p` is small, and we want theta_transpose * x = p * norm theta to be either >= 1 or <= 1, then in both cases theta has to be large.
- For a good decision boundary
	- **If the decision boundary is at 90 degree angle** to most examples in the training set, the lengths of the projections are much bigger (maximum, actually).
	- Since we are still subject to the same constraints for theta_transpose * x = p * norm_theta, this means the SVM will optimise **theta to be smaller**
	- This is what we want - we want smaller values for the parameters.
	- To maximise projections, the margin between the clusters of training examples and the classification boundary must be large.
- This entire derivation was done assuming theta_0 = 0, which meant we were constraining decision boundary hypothesis space to lines that pass through origin.
- But even without this constraint, the SVM works in exactly the same way: it maximises the margins separator betweeen positive and negative examples. 

## Kernels 1
- The technique for adapting SVMs to develop complex non-linear classifiers.
- Suppose your dataset consists of two classes that are **not linearly separable**.
	- One way of developing a non-linear decision boundary is to use a high order polynomial.
	- New notation: theta_0 + theta_1 * f_1 + theta_2 * f_2 + ... + theta_n * f_n
		- f_1, f_2, f_3,..., f_n are new features e.g. f1 = x1, f2 = x2, f3 = x1* x2, f_4 = x1^2,../
- But is there a different/better/less computationally expensive choice of features?
- As an example, consider a 2D space of features spanned by `x1` and `x2` and identify **landmarks** l1, l2, l3 and compute new features depending on proximity to landmarks. 
	- f_i = Similairty(x, l_i) = exp(-||x - l_i||^2 / 2 * sigma^2)
	- This is a **Kernel Function** - specifically, the **Gaussian Kernel Function**.
- If the point `x` is close to or similar to landmark `l_i`, then the similarity function's output is approximately exp(- 0 /sigma^2) ~ 1.
- Similarly, if the point `x` is far from the landmark, the similarity function output is ~ 0.
- Each landmark `l1`, `l2`, and `l3` defines a **new feature** - f_1, f_2, f_3.
- Plotting similarity function for two features in `x`
	- Z axis shows how similar specific values of a feature will be in Gaussian feature or landmark `l1`.
	- Sigma_squared
		- Smaller: kernel looks similar, but width of the contours decreases. The similarity falls from 1 to 0 more rapidly as we move away from landmark.
		- Larger: kernel looks similar, but is spread ver a larger area: the similarity decreases much more slowly as we move away from the landmark. 
- Based on the new features derived by the kernel/similarity functions, we have a new hypothesis
	- Predict 1 if theta_0 + theta_1 * f_1 + theta_2 * f_2 + theta_3 * f_3 >= 0
	- Supposed we've already learnt that the theta_0, 1, 2, and 3 are 0.5, 1, 1, and 0.
		- For a point close to l_1, theta_1 * f_1 will be large, whereas theta_2 * f_2 and theta_3 * f_3 will be smaller. However, sum is > 0, we predict y = 1.
		- For a different point (far away from l1, l2, and l3), f_1, f_2, and f_3 are all going to be closer to 0, so we will predict y = 0.
	- Find that we predict positive for points close to l1 and l2, and negative class for points far away from them. This helps us draw a decision boundary.  
- **So as part of using SVMs, we will often use kernel functions of similarity functions to find decison boundaries based on a hypothesis function that uses landmarks to compute new features.**

## Kernels 2
### Choosing Landmarks
- Where to get landmarks from? Should we choose more landmarks for more complex problems?
- Suppose we put landmarks at exactly the same locations as the training examples in `n` dimensional feature space.
	- l_1, l_2, l_3,...,l_m: one landmark per `m` locations in the feature space.
	- This is nice because it says the features are going to measure how close a given example is close to examples already present in the training set. 
	- So given (x1, y1), (x2, y2), (x3, y3),...,(xm, ym) choose l1 = x1, l2 = x2, l3 = x3,..., lm = xm; (can be in training/test/validation sets).
	- And then we create a set of new features f1, f2, f3,..., fm.
- Given x_i, we would map it to f1_i = sim(x_i, l1), f2_i = sim(x_i, l2), f3_i = sim(x_i, l3),...
- Somewhere in this list, at the `i`th component, we will have one special feature component fi_i = sim(x_i, l_i) = exp(-0) = 1. 

### Using SVM
- Compute the features `f` which will be an `m + 1` dimensional feature vector (one feature per landmark, one landmark per trainin example +  f_0 = 1)
- Predict y = 1 if theta_transpose * f >= 0 
- Get parameters theta using the SVM cost function, but with x_i replaced with the kernel features f_i.
	- The number of features we have is the same as the number of training examples (since one feature per training example) => `n` = `m`.
	- Still not regularizing the parameter theta_0
- The regularization term is implemented differently in SVM. Not implemented as theta_tranpose * theta, but rather theta_transpose * M * theta
	- This allows the SVM to scale to much bigger training sets and allows optimisation to be faster.
	- Not the same as square norm of Z, but scaled by a matrix `M`. Andrew Ng did not define what this matrix is. 
	- Implementation detail: does not change the conceptual core of the SVM algo. 
- Why not apply kernel to logistic regression?
	- Can actually do this for logistic regression, but the computational tricks don't generalize well to these other algorithms.
	- So actually slows down these algos. 
	- SVMs and Kernels go together, but other algos do not.

### Choosing SVM Parameters
- One of the parameters in SVM is `C` of the optimisation objective (played a role similar to 1 / lambda - the regularization param of log reg).
	- Large C: low bias, high variance, more prone to overfitting. 
	- Small C: high bias, low variance, more prone to underfitting.
- Choosing the parameter `sigma^2`.
	- If large, then in similarity function/gaussian function falls off slowly.
		- Smoother function, features `f_i` vary more smoothly, higher bias and low variance.
	- Smaller value means that the gaussian function falls off rapidly.
		- Not a smooth function, features `f_i` will vary less smoothly.
		- Lower bias, higher variance. 

## Using an SVM
- Don't reinvent the wheel: use a library or SVM software package, just as you would for inverting a matrix or finding square root.
- SVM software packages such as `liblinear` and and `libsvm` are useful for efficiently finding the parameters `theta`.
- But you still need to specify
	- Parameter `C`: tradeoff between bias and variance
	- Choice of kernel or similarity function
		- **No kernel**: a linear kernel i.e. predict y = 1 if theta_transpose * x >= 0.
			- Used when `n` is large, and `m` is small
			- Small dataset, so easier to fit a linear decision boundary instead of attempting to learn a complex, higher dimensional boundary.
		- **Gaussian kernel** 
			- Will have to choose the parameter `sigma^2`
			- If `n` original features and `n` is small, and `m` is large (e.g. 2D dataset of 100s or 1000s of examples), can use the Gaussian kernel to fit a complex, non-linear decision boundary. 
			- **Do perform feature scaling before using the Gaussian kernel.**
			- SVM library will automatically generate features from a vector of original features, sometimes without us having to specify the functional implementation that maps the original features to new ones. 
- Gaussian and Linear kernels are the most commonly used similarity functions. 
- Not all functions can be similarity/kernel functions: need to satisfy Mercer's theorem, and must be amenable to optimization, and should not diverge. 
	- Polynomial
		- (x_tranpose.l)^2: large dot product if the two are similar
		- (x_transpose.l + 1)
		- (x_transpose.l + 5)^4
		- So has two parameters: the **constant** added to the inner product and the **degree** to which the dot product will be raised. 
		- Used only for data when `x` and `l` are strictly non-negative. 
	- String kernel
		- Input data is text/string.
	- Chi-square kernel
	- Histogram kernel 
	- Intersection kernel

### Multiclass Classificate
- Many SVM packages already have multiclass classification functionality built into them.
- Can also use a **one vs. all** approach
	- Train `K` SVMs, one per each distinct class, that distniguishes one class from all other classes.
	- This gives theta_1, theta_2, theta_3, ..., theta_k
	- Pick the class `i` with the largest (theta_i_transpose * x)

### LogReg vs SVM: When to use what?
- If `n` >> `m` e.g. more features than training examples, use logistic regression or **SVM with linear kernel**
- If `n` is small but `m` is intermediate (e.g. n = 1 - 1000, m = 10 - 10000)
	- Use SVM with Gaussian kernel 
- If `n` is small but `m` is large (e.g. n = 1 - 1000, m = 50k - millions)
	- SVM with Gaussian kernel will be slow to run (as of 2011/2012).
	- Manually create more features and then use logistic regression or SVM without a kernel. 
- Logistic regression and SVM without a kernel usually give similar performance, but depending on implementation details one may be slightly more efficient than the other. 
- Where do neural networks fit in?
	- A well designed NN is likely to work for all these cases.
	- But may be slower to train than an SVM for some of these cases. 
- SVM has a convex optimization problem, so SVM packages will **always** find the global minima instead of local minima when optimising cost.
- Local optima don't tend to suffer too much from local optima, but if this happens, SVMs are likely to work well. 