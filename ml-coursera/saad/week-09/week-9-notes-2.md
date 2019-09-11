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

### 