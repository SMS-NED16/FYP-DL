#  BUILD WDNN MODEL
from tensorflow import keras


def build_model(wnn_activation='relu', wnn_units=60, cnn_activation='relu',
               cnn_filters=8, cnn_kernel_size=(3, 3), cnn_pool_size=(3, 3), 
               cnn_dense_activation='relu', optimizer='adam', show_summary=False,
               show_graph=False): 
  """Function that builds a WDNN with the specified hyperparameters. Will be passed
  as a build function to a `kerasClassifier` wrapper"""
  # Defining some constants - bad practice, need to find a way to do this automatically
  num_features = 1034 
  weeks_per_consumer = 148
  days_per_week = 7
  channels = 1

  # INPUT LAYERS
  wnn_input = keras.layers.Input(name='wnn_input', shape=(num_features,))
  cnn_input = keras.layers.Input(name='cnn_input',
                                 shape=(weeks_per_consumer, days_per_week, channels))
  
  # WNN HIDDEN LAYERS
  wnn_dense_1 = keras.layers.Dense(activation=wnn_activation, units=wnn_units)(wnn_input)


  # CNN HIDDEN LAYERS
  # First convolutional layer
  cnn_conv_1 = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation, 
                                    padding='same', kernel_size=cnn_kernel_size)(cnn_input)

  # Second Conv2D
  cnn_conv_2 = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation, 
                                    padding='same', kernel_size=cnn_kernel_size)(cnn_conv_1)

  # Third Conv2D
  cnn_conv_3 = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation, 
                                    padding='same', kernel_size=cnn_kernel_size)(cnn_conv_2)

  # Fourth Conv2D
  cnn_conv_4 = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation, 
                                    padding='same', kernel_size=cnn_kernel_size)(cnn_conv_3)

  # Fifth Conv2D
  cnn_conv_5 = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation,
                                    padding='same', kernel_size=cnn_kernel_size)(cnn_conv_4)

  # CNN Max Pooling
  cnn_max_pooling_2d = keras.layers.MaxPooling2D(pool_size=cnn_pool_size)(cnn_conv_5)

  # Fully connected classifier
  cnn_flatten = keras.layers.Flatten()(cnn_max_pooling_2d)
  
  # Number of units must be same as wnn_dense_1 if usign `add`, `subtract`, `multiply`, `dot` for merge 
  cnn_dense_1 = keras.layers.Dense(units=wnn_units, 
                                   activation=cnn_dense_activation)(cnn_flatten)

  # Merge CNN and WNN outputs
  merged_outputs = keras.layers.add(inputs=[wnn_dense_1, cnn_dense_1])

  # Output
  wdnn_output = keras.layers.Dense(units=1, activation='sigmoid', name='main_output')(merged_outputs)

  # Build the computation graph and return it 
  wdnn_model = keras.models.Model(inputs={'wnn_input': wnn_input, 'cnn_input': cnn_input}, 
                            outputs={'wdnn_output': wdnn_output})
  
  # Compile it with the optimizer
  wdnn_model.compile(loss='binary_crossentropy', metrics=[keras.metrics.AUC()], 
                     optimizer=optimizer)
  
  # Optionally, print summary and show model graph
  if show_summary:
    print(wdnn_model.summary())
  
  # Optionally, show an image of the model's architecture
  if show_graph:
    keras.utils.plot_model(wdnn_model, show_shapes=True, show_layer_names=True, 
                           to_file='model.png')

  # Return wdnn_model
  return wdnn_model


# TODO: Make WDNN_Model_Builder class. Too much code duplication because of these functions
def build_wdnn_n_conv(n_conv=5, wnn_activation='relu', wnn_units=60, cnn_activation='relu',
               cnn_filters=8, cnn_kernel_size=(3, 3), cnn_pool_size=(3, 3), 
               cnn_dense_activation='relu', optimizer='adam', show_summary=False,
               show_graph=False):
	"""
	Variant of the original build_WDNN function that uses a for loop to add identical
	2D convolutional layers.

	Assumes at least one Conv2D layer will always be added.  
	"""
	num_features = 1034 
	weeks_per_consumer = 148
	days_per_week = 7
	channels = 1

	# INPUT LAYERS
	wnn_input = keras.layers.Input(name='wnn_input', shape=(num_features,))
	cnn_input = keras.layers.Input(name='cnn_input',
	                             shape=(weeks_per_consumer, days_per_week, channels))
  
	# WNN HIDDEN LAYERS
	wnn_dense_1 = keras.layers.Dense(activation=wnn_activation, units=wnn_units)(wnn_input)


	# CNN HIDDEN LAYERS - added with a for loop

	# Assuming one Conv2D layer will always be added to the model 
	assert n_conv >= 1
	cnn_conv_1 = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation, 
	                                padding='same', kernel_size=cnn_kernel_size)(cnn_input)

  	# Remember last convolutional layer added
	last_conv_layer = cnn_conv_1 

	for i in range(1, n_conv):
		# Add new conv layer that takes previous conv layer's output tensor as input
		new_conv_layer = keras.layers.Conv2D(filters=cnn_filters, activation=cnn_activation, 
			padding='same', kernel_size=cnn_kernel_size)(last_conv_layer)

		# Update the last convolutional layer added
		last_conv_layer = new_conv_layer

	# CNN Max Pooling
	cnn_max_pooling_2d = keras.layers.MaxPooling2D(pool_size=cnn_pool_size)(last_conv_layer)

	# Fully connected classifier
	cnn_flatten = keras.layers.Flatten()(cnn_max_pooling_2d)

	# Number of units must be same as wnn_dense_1 if usign `add`, `subtract`, `multiply`, `dot` for merge 
	cnn_dense_1 = keras.layers.Dense(units=wnn_units, 
	                               activation=cnn_dense_activation)(cnn_flatten)

	# Merge CNN and WNN outputs
	merged_outputs = keras.layers.add(inputs=[wnn_dense_1, cnn_dense_1])

	# Output
	wdnn_output = keras.layers.Dense(units=1, activation='sigmoid', name='main_output')(merged_outputs)

	# Build the computation graph and return it 
	wdnn_model = keras.models.Model(inputs={'wnn_input': wnn_input, 'cnn_input': cnn_input}, 
	                        outputs={'wdnn_output': wdnn_output})

	# Compile it with the optimizer
	wdnn_model.compile(loss='binary_crossentropy', metrics=[keras.metrics.AUC()], 
	                 optimizer=optimizer)
  
	# Optionally, print summary and show model graph
	if show_summary:
		print(wdnn_model.summary())

	# Optionally, show an image of the model's architecture
	if show_graph:
		keras.utils.plot_model(wdnn_model, show_shapes=True, show_layer_names=True, 
	                       to_file='model.png')

	# Return wdnn_model
	return wdnn_model