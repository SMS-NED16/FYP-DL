k = 4				# number of folds or sections data will be divided into

# Validation set samples = 1/ k of the total samples
num_validation_samples = len(data) // k  

# Shuffle the data to account for effect of variance - this is done only once
np.random.shuffle(data)

# Create a list to store the validation set scores for each model
validation_scores = []

# For every fold
for fold in range(k):
	# Select the validation data partition
	validation_data = data[num_validation_samples * fold : 
	num_validation_samples * (fold + 1)]

	# Use the remainder of the data as training data
	# The + in this statement is CONCATENTATION, not summation
	traning_data = data[:num_validation_samples * fold + 
	* num_validation_samples * (fold + 1):]

	# Instantiate a new model to train/evaluate with the given train/validation data
	model = getModel()
	model.train(training_data)
	validation_score = model.evaluate(validation_data)
	
	# Append the current model's evaluation scores to the list
	validation_scores.append(validation_score)

# The final validation score is the average of validation scores
# for all other data points - we can tune our model's hyperparams based on this score
avg_validation_score = np.average(validation_scores)

# After hyperparameter tuning based on avg validation scoore
model.train(data)							# train on the entire data - no validation hold out
test_score = model.evaluate(test_data)		# evaluate final performance on the test set