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