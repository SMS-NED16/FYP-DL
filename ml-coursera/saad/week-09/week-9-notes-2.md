# Coursera - Machine Learning - Week 9 - 2: Recommender Systems

## Recommender Systems - Problem Formulation
- An important application of ML: many SV people are trying to build better recommender systems.
	- FAANG all use recommender systems: responsible for generating a substantiatl fraction of their revenue.
	- Receives relatively little attention in academia, but very important in industry.
- No feature engineering
	- Features we choose have a big effect on the performance of the learning algorithm.
	- For some problems, algos can attempt to automatically learn good features for you e.g. neural networks. 
	- Recommender systems are an example of a self-tuning system and require little feature engineering.

## Example - Predicting Movie Ratings
- We're a website that sells or rents movies to users and allow them to rate movies (0 - 5).
- Suppose we have 4 users who have rated some (or all) of the 5 movies in our database.
- Notation
	- `n_u` = number of users in the dataset
	- `n_m` = umber of movies in the dataset
	- `r(i, j)` is 1 if the `i`th movie has been rated by user `j`.
	- Whenevr r(i, j) = 1 = 1, we also get `y(i, j)` 
		- is a number from 0 - 5 that shows the rating user `j` gave to movie `i`.
- The problem is, given the dataset, to predict the ratings that a specific user will give for a specific movie that **they have not yet watched or rated**.
- We can then use the predicted rating for a given user on a given movie to recommend that movie to the user.

## Content-Based Recommendation Systems
- Our first appraoch to building a recommendation system.
- We create two features:
	- x1: the extent to which the movie is a romantic movie. (0.0 - 1.0)
	- x2: the extent to which the movie is an action movie. (0.0 - 1.0)
- With these features, each movie can be represented as a feature vector which encodes the probability that it belongs to a specific genre.
- `n` is the number of features.
- We can treat predicting the rating for a given movie by a given user to be a regression problem.
	- For each user `j`, learn a parameter `theta^j` that is in `R^3`.
	- Predict user `j` as rating movie `i` with `(theta^j)transpose.x^(i)` stars.
- Each user has a separate parameter vector `theta_1, theta_2, theta_3, ..., theta_n` respectively. 

### Problem Formualtion
- `r(i, j)` = 1 if the user has rated the movie `i` (0 otherwise).
- `y(i, j)` = rating on user `j` on movie `i` if defined
- `theta^j` is the parameter vector for the user `j`
- `x^i` iis the feature vector for movie `i`.
- For the user `j`, movie `i`, predicted rating is `(theta^j)^T.(x^i)`.
- `m^j` is the number of movies rated by user `j`.
- To learn `theta_j`, we can use linear regression: see handwritten notes.
- Looks similar to the linear regression gradient descent except that `m` has been absorbed as a constant.
- Called content-based approach because we assume we already have features about the content of the movies to help make predictions.

## Collaborative Filtering Recommender Systems
- Can be difficult/time-consuming/expensive to have every user watch every movie and rate it as belonging every possible category.
- Suppose that we have a dataset where we know the movies and which movies have been rated by which users (along with their ratings), but no data about how much a specific movie belongs to a specific category.
- But we do know how much a user likes a specific genre of movies: theta_1, theta_2, theta_3, theta_4,..., theta_nu.
- It becomes possible what the values of `x1` and `x2` will be for each movie given a user. 
- What feature vector should `x_i` be so that `theta_j times x^i`.
- Usually, in regression problems, we would use gradient descent to find the parameters that would minimise the cost function.
- In this case, we are attempting to find the **features** that will mini

### Optimization Algorithm
- Given `theta_1, theta_2, theta_3, ..., theta_nu` we want to learn `x_i`.
- This is opposite to the conventional supervised learning approach we've seen so far where we use features `x_i` to predict the parameters `theta` that will lead to the best I/O mapping.
- We're trying the optimal parameters `x_i` that will lead to a predicted score for a movie `theta_transpose x_i` that is as similar to the actual rating `y` for each rated movie rated by each user.


### Collaborative Fultering
- Given a vector of features `x` we can learm a vector of parameters `theta`.
- Alternatively, given a vector of parameters `theta` we can learn a vector of features `theta`.
- Chicken and egg problem: which comes first?
	- Can start with a random set of features and learn `theta` approximation 1.
	- Then use `theta` to learn a better set of features `x`.
	- Then repeat.
	- Back and forth between learning `theta` and then learning `x` will cause the algorithm to converge to a reasonable set of featueres and parameters.
- Guess theta > x > theta > x > theta > x and so on and so forth until convergence.
- This is possible for this problem only because each user rates multiple movies and that allows us to iterate back and forth between the features and parameters.
- So when we run the algo on a large set of users, all users are effectively collaborating to help the algorithm learn features that can be used by the system to make better recommendations for all other users.

## Optimising Collaborative Filtering
- Don't need to iterate back and forth between learning features and learning parameters.
- Can optimise the cost function as a parameter of both the features and parameters while also regularizing both of them. 
- Allows us to solve for optimal `theta` and `x` simultaneously using a new optimisation objective. 
- First optimisation summation is sum over all users `j` and all movies rated by that user.
- Second optimisation summation does the same thing but in the opposite order: for all movies `i` and all user ratings for that movie
- So the first term in the new optimisation objective is simply combining both of those summation terms. 
- It also has two regularization sums: one for parameters, and one for features. 
- It's a mutlivariate optimisation/differentiation process.
- The only difference is that we are no longer going back and forth between minimising only features and minimising only parameters. 

### Regularization
- Previously, during regularization of features we assumed the dummy or bias feature was always 1, but this time we're getting rid of `xo`.
- So the feature and parameters will be vectors in `R_n`.
- The reason we're doing this is because we're learning all the features, so there is no need to hardcode a feature that will be equal to one. If the algo needs to learn a feature that is equal to one, it now has the flexibility to do so itself. 

### Algorithm
- Initialize `x` and `theta` to small random values. 
- Minimise the cost function using gradient descent or an advanced optimisation algorithm.
	- See handwritten notes/lecture slides.
	- No longer have `x_0` = 1 so both `x`, and `theta` are `n` dimensional vectors, not `n + 1` dimensional.
	- No separate update for `x_0` and `theta_0`.
- Given a user with some parameters `theta` and some movie with learned features `x`, we can predict that the user will rate the move `theta_transpose * x`.