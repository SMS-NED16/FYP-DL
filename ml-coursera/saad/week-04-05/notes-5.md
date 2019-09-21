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