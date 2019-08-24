# Define the number of samples that will be used for validation
num_validation_samples = 10000

# Randomly shuffle the data prior to train-test-validation split
np.random.shuffle(data)

# The validation data consists of all samples up to (but not including)
# the index equal to the num_validation_samples
# This means the first 10k samples from the randomly shuffled dataset is for validation
validation_data = data[:num_validation_samples]

# The remaining samples are the training data
data = data[num_valiation_samples:]

# Slicing the modified data array - why?
training_data = data[:]

# Instantiate a model
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# At this point, you can tune your model, retrain, re-evaluate, and retune

# Once the model's hyperparameters have been tuned, it's common to 
# train your final model from scratch on all the non-test dat available
# This means we can combine the training data with validation data for training
# More training samples - less chance of overfitting - better model
# Also, this doesn't drastically modify the parameters or performance because the 
# Model has been implicitly tuned to fit the validation_data anyway through hyperparams 
model = get_model()
model.train(np.concatenate(training_data, validation_data))

# Final evaluation on the test data 
test_score = model.evaluate(test_data)


"""Drawback - if dataset is small, then validation and test sets
may have too few samples to be statistically representative of the data at hand.
If different random shuffling before splitting yields different values for 
accuracy/measures of performance, then this problem is occurring."""