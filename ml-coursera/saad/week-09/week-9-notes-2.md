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