# Coursera - Week 4 - Neural Networks
- An old idea that temporarily went out of style but has recently become very popular.
- Why do we need neural networks when we already have other learning algorithms?
- Sometimes we need to learn non-linear hypotheses, and this is difficult to do with conventional shallow learning approaches. 

## Non-Linear Hypothesis
- Applying logistic regression to non-linear decision boundaries.
	- Have many polynomial features e.g. x1 * x2, x2 * x3, x1^2, x2^2.
	- But this only works well when we have few features.
- Suppose instead that we're attempting to solve a more complex problem e.g. predicting housing price using 100 features.
	- In this case, using polynomial features becomes computationally intractable.
	- If we include all the combinations of features just to create second order terms, we have 5000 features.
	- The number of quadratic features grows as O(n^2) / 2
		- Higher risk of overfitting
		- Computationally intractable.
	- Could try including a subset of these features. 
		- Instead of creating a hypothesis that uses all 5000 combinations of quadratic features, use just 100 quadratic features.
		- Can fit axis-aligned ellipses or circles, but not more complex decision boundaries.
	- If we were to try and fit higher order polynomial features (e.g. cubic features)
		- 170k different features
		- Not a good way to build non-linear classifiers.
- This is relevant because for most machine learning problems, the number of features `n` is usually very large.
- Consider the example of computer vision
	- Computer sees image as a matrix or grid of pixel intensity values.
	- CV problem is to look at several such matrices (the labeled training set - cars and not cars) and learn to differentiate the two. 
	- This requires a complex non-linear hypothesis. 
	- Look at the same two pixel locations (x1, x2) in different pictures (positive and negative examples). This is done for all examples, and we find that cars and non-cars lie in different regions of the feature space and are not linearly separable. 
	- If we considered only 50 x 50 px pictures, we would have 2500 pixel intensity values (features) 
		- 7500 if RGB because one pixel intensity value per each channel.	
	- If we tried to learn the non-linear hypothesis using only all the quadratic features: **3 MILLION FEATURES**.
		- Inefficient and computationally intractable.

## Neurons and the Brain 
- Neural networks were motivated by the idea of building machines that can mimic the brain. 
- However, even though they are biologically motivated, they are a mathematical, rather than biological, model.
- Origins
	- Algorithms that try to mimic the brain - the most amazing learning machine we know about.
	- Used throughout 80s and 90s, but popularity diminished (due to hardware limitations).
- Recent resurgence: state-of-the-art technique for many applications, popular because computers are now finally fast enough to carry out computations for neural networks.

### One Learning Algorithm Hypothesis
- The brain uses a single mechanism to learn a wide variety of skills: seeing, discerning voice, smell, taste, motion, etc.
- Neural scientists have carried out neural rewiring studies:
	- Signals from the eyes routed to auditory cortex (used for listening): animals learn to 'see' or perform visual recognition
	- Signals from the eyes routed to somatosensory cortext (used for touch): animals learn to 'see' using touch.
- So instead of learning to implement a thousand different algorithms to do different tasks, we should implement an approximation of the one learning algorithm used by brains that can then generalize to different tasks. 

### Sensor Representations in the Brain
- BrainPort: camera takes grayscale image of what's in front of you, maps pixel intensity values to an array of electrodes, array placed on tongue, humans eventually learn how to use this input to see. 
- Human Echolocation: Visually impaired people use pattern of sounds reflected off surfaces to map surroundings and move around.
- Haptic Belt: Northmost buzzer always buzzing -> human learns to use this as input to figure out direction, just like birds do.
- Implanting 3rd eye in frog.

## Model Representation - Part 1
- NNs were built as simulations of neurons or nerve cells in the brain.
- Each neuron has a cell body, a number of input wires (dendrites - receive signals from other neurons), and an output wire (axon - send outputs to other units).
- Neurons communicate by sending little pulses of electricity: dendrites connect to axons.
- In NNs, the building block is thus a neuron or artificial neuron that acts as a single logistic unit. 
	- Takes inputs x0, x1, x2, x3,... from input wires and outputs a hypothesis using h_theta(x) = 1/(1 + e^(-theta_transpose * x)) 
	- Inputs are X, parameters are THETA aka weights.
	- x_0 is the bias unit and is always 1
	- The neuron is said to have a sigmoid or logistic activation function - the non linearity `g(z)` = 1/(1 + e^-z)
- A neural network is a group of these neurons strung together.
- First layer is called the input layer.
- Final layer is called the output layer - it has the neuron that outputs the final value computed by the hypothesis.
- Any layers between the input and outpout layers are acalled the hidden layers. 
	- Processes values that are not in our training set.
	- Inputs to units in these layers are not exactly `x`, and their outputs are not exactly `y`. They're hidden.
- If a layer network has s_j units in layer j and s_(j+1) units in layer (j + 1), then the matrix of weights mapping the transformation from layer j to layer j+1 will be a (s_(j+1), s_j + 1) dimensional matrix. 

## Model Representation - Part 2
- The argument for each activation is a weighted linear combination of the outputs from the previous layer.
- This allows us to vectorize the activation computation.
- The process of computing h_theta(x) is called **Forward Propagation**
	- Computing activations for successive layers in the NN by passing on or propagating previous layers activations to the next layer.

