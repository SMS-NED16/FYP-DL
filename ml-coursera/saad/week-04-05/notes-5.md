# Coursera - Machine Learning - Week 5

# Neural Networks - Part 2

## Cost Function
- Crucial to help design a learning algorithm which helps a neural network optimise its parameters.
- Suppose we have a multiclass, 4 layer classification neural network. 
- `L` = total number of layers = 4
- `s_l` = number of neurons or units (**not counting the bias unit**) in layer `l` of the network.
	- For binary classification, only 1 output unit that computes h_theta(x) which is a real number => k = 1
	- For multiclass classification, k output units - one per each class - which toghether compute h_theta(x) => k-dimensional vector.
	- Usually k >= 3 since if 2 classes, can just use one unit with one-vs-all method.
- Cost function is generalization of the one used for logistic regression.
- See handwritten notes.

## Backpropagation
- The algorithm that uses the cost to optimise the parameters. 
- Allows us to compute the partial derivatives of the cost with respect to the weights.
- See handwritten notes.
- Backpropagation is neural network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression.
- To minimize J(theta), we need to compute the partial derivatives of the cost J(theta) with respect ot all of weight mapping the activation of the jth unit in layer i to the ith unit in layer (j + 1).
- The capital D matrix is an accumulator - adds up our values as we go along and is eventually used to compute our partial derivative. 


## Autonomous Driving
- A fun and historically example of neural networks: teaching cars to learn to drive themselves.
- Prior to training, the NN has no idea about how to steer the car, so the entire steering band is uniformly shaded: equally likely to steer in any random direction.
- As the human drives, the NN learns to map changes in the direction in which the user steers and the way in which the road's direction changes.
- 3 layer network uses backpropagation to learn the same steering direction as the driver from a 32 by 30 px image.
- Learns to imitate human driver in **2 minutes of training**.
- Same training procedure is repeated for other road types. 
- Multiple NNs output a steering direction - the output of the **most confident** network is used. 
- As the vehicle approaches an intersection, confidence of the lone-lane network decreases.
- Confidence of two-lane network rises when it detects 2 lane networks. 

## Unrolling Parameters
- When implementing backpropagation, we will have to unroll our matrix of parameters into a vector.
- Consider a function
function [jVal, gradient] = costFunction(theta) 
...theta, gradient are all R_(n + 1) dimensional vectors
	// Computes both the cost and the gradient
- Can pass this function as an argument to an advanced optimisation function like `fminunc`
	- All of them take a cost function as an argument along with initial values of the parameters
	- Will then return the optimised parameters for our model.  
`optimisedTheta = fminunc(@costFunction, initialTheta, options)`. 
- For a neural network with 4 layers
	- theta_1, theta_2, theta_3 - matrices of parameters mapping transformations from one layer to the next 
	- D1, D2, D3 - accumulator matrices representing sum of derivatives for params in a specific layer.
- All these matrices need to be unrolled into vectors so that they can be passed as arguments to an optimisation function.
- Concatenate these elements in MATLAB
thetaVec = [theta_1(:); theta_2(:); theta_3(:)];
DVec = [D1(:); D2(:); D3(:)];
- To revert back from vectors to matrices
Theta1 = reshape(thetaVec, (1:110), 10, 11);		// first 110 elements are for theta 1 - into 10 x 11 matrix
Theta2 = reshape(thetaVec, (111:220), 10, 11);		// the next 110 elements are for theta 2 - into 10 x 11 matrix
Theta3 = reshape(thetaVec, (221:231), 1, 11);		// the last 11 elements are theta 3 - reshaped into 1 x 11 vec

### Learning Algorithm
- Have initial params theta_1, theta_2, theta_3. 
- Unroll to get initialTheta to pass to fminunc(@costFunction, initialTheta, options).
- Write a function to compute cost and gradient function [jVal, gradientVec] = costFunction(thetaVec)
	- From thetaVec, get theta_1, theta_2, theta_3.
	- Use forward propagataion and back propagation to get D_1, D_2, D_3, and J_theta.
	- Unroll D_1, D_2, D_3 to get gradientVec

## Gradient Checking
- Backprop is complicated/very detailed/difficult to implement: many ways to have subtle bugs. 
	- Could look like its working fine with a specific optimisation algorithm, but performance would actually be worse than it should be.
- Solution: gradient checking - a way of verifying that gradient descent has been implemented correctly.
- Verifies that code does indeed compute derivative of the cost function J.
- Numerically approximate the derivative at a point by
	- Compute J(theta + epsilon) = y1
	- Compute J(theta - epsilon) = y2
	- Compute slope between ((theta - epsilon, y1) and (theta + epsilon, y2)).
	- Gradient at this point = (y2 - y1) / 2 * epsilon
	- Epsilon is usually very small. 10^-4 or -5.
		- The smaller the value of epsilon, the better the approximation of the derivative, but the more expensive the computation. 
	- This is the **double-sided difference** formula. Gives a more accurate estimate of the slope than the single sided estimate.
- Implement a function that calculates the gradient at a specific theta and specific value of epsilon.
`gradApprox = (J(theta + EPSILON) - J(theta - EPSILON))/(2 * EPSILON)`

### What if `theta` is a vector?
- Theta may not be a real number, but an `n` dimensional vector (after unrolling).
- Compute approximations for the partial derivatives of the cost function with respect to the `n`th component of `theta` in the same manner - double sided, add and subtract epsilon.
- Implementation in Octave
`for i = 1:n
	thetaPlus = theta;							// unrolled version of all parameters
	thetaPlus(i)  = thetaPlus(i) + EPSILON;		// add epsilon to the ith component
	thetaMinus = theta;							// do the same for the minus
	thetaMinus(i) = thetaMinus(i) - EPSILON;
	gradApprox(i) = (J(thetaPlus) - J(thetaMinus) / (2 * EPSILON))
end;`
- We will check that `gradApprox` is approximately the same as `DVec` - the derivatives we got from `backprop` function.
	- Approximately means we must specify a threshold: how close must the two values be (up to a few decimal places?) for us to be confident that implementation of backpropagation is correct?

### Implementation Note
- Implement backprop to compute `DVec` (unrolled D_1, D_2, D_3, ...)
- Implement numerical gradient check to compute `gradApprox`.
- Make sure they give similar values.
- Turn off gradient checking - use backprop code for learning.

### Important
- Be sure to disable gradient checking code before training classifier.
- Running numerical gradient computation on every iteration of gradient descent will cause code to be **very slow**.
- This is a very computationally expensive way to try and approxmimate the derivative. 
- Backprop is more computatinally efficient wawy to compute derivatives.
- We don't want to use numerical computation for actually computing derivatives - we just want to check that the more efficient backprop gives the same results (correct results) as the numerical method.

## Random Initialization
- Need to pick some initial values for `theta` for gradient descent or the other advanced optimization algorithms.
- Is it possible to set the initial value of theta to all zeroes?
	- Not a good idea for neural network.
	- Both hidden units will be computing the same function of the inputs, and will end up with a2_1 = a2_2 for all training examples. 
	- Since the outgoing weights will be the same, the delta values will also be the same. 
	- This will cause all the weights of the parameter to be equal to each other, even after each update.
	- **After each upate, parameters corresponding to inputs going into each of two hidden units are identical**.
	- So even after one iteration of gradient descent, the two hidden units will still be computing the same function of the inputs.
	- So NN can't compute interesting functions - hypothesis space is limited. 
	- All hidden units would be computing the exact same feature, which would be highly redundant.
- The solution is **random initialization**.
	- It solves the problem of **symmetrical weights**.
	- Init to a random value between [-epsilon, +epsilon]
	- `theta_1 = rand(10, 11) * (2 * INIT_EPSILON) - INIT_EPSILON` // (10, 11) is shape of matrix
	- `theta_2 = rand(10, 11) * (2 * INIT_EPSILON) + INIT_EPSILON` // (1, 11) is shape of vector - 11 elements
	- **This epsilon has nothing to do with the gradient checking epsilon**

## Putting It Together
- Pick a network architecture (connectivity pattern between neurons).
	- Vary hidden units, layers, and I/O units.
	- Number of input units = dimensions of the features `x_i`.
	- Number of output units = number of classes (in case of multiclass classification)
	- Using a single hidden layer is a reasonable default, or to have the same number of hidden units in each hidden layer. 
	- The more hidden units we have, the better the model's performance, but the slower it is to train.
		- Usually comparable to the number of features in `x` (sometimes slightly more).
- Train a neural network
	1. Randomly initialize weights to small values (near zero).
	2. Implement forward propagation get h_theta(x_i) for any x_i.
	3. Implement code to compute cost function J_theta.
	4. Implement backpropagation algorithm to compute partial derivatives with respect to the parameters
	`for i = 1:m
		Perform forward propagation and backpropagation using example (x_i, y_i).
		Get activations a_l and delta terms delta_l for l = 2, 3, 4,..., L.
		Delta_l = Delta+l + delta_(l+1) * (a_L)transpose
	`
	Compute the partial derivative terms while accounting for the regularization parameter lambda.
	5. Use gradient checking to compare partial derivative terms computed using backprop with those computed using numerical methods.
	6. Disable gradient checking code if partial derivative terms verified.
	7. Use gradient descent or advanced optimization method with backpropagation to try and minimize J_theta as a function of parameters theta. 
- For NNs, cost function is non-convex and so is susceptible to local minima, but it turns out in practice this is not usually a huge problem.
- So even though the global optima is not guaranteed with GD, it doesn't really matter. 