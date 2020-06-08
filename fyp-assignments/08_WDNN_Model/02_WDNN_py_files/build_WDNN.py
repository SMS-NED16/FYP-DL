#  BUILD WDNN MODEL
from tensorflow import keras

def build_model(wnn_activation='relu', wnn_units=54, cnn_activation='selu', 
  cnn_filters=[8], cnn_kernel_Size=(3, 3), cnn_pool_size=(3, 3), cnn_dense_activation='relu', 
  optimizer='adam', show_summary=False, show_graph=False):
  """
  build_WDNN_model - builds and returns a Wide and Deep Convolutional Neural Network with the 
  specified hyperparameters using Keras Functional API.


  INPUTS
  - `wnn_activation` (str): Activation function for WNN dense units. 
  Choose from ['relu', 'selu', 'softmax', 'tanh', 'sigmoid']
  - `wnn_units` (uint): Number of units in dense layer of WNN component. 
  - `cnn_activation` (str): Activation function for Conv2D layer in CNN component. Choose from
  ['relu', 'selu', 'softmax', 'tanh', 'sigmoid']
  - `cnn_filters` (list/iterable): A list that defines the number of filters in each Conv2D layer.
  One Conv2D layer is created for each entry in this list, so the length of the list determines the
  number of Conv2D layers. [8, 12, 16] will create three consecutive Conv2D layers with 8, 12, and 16 
  filters respectively.
  - `cnn_kernel_size` (tuple of integers): Window size for Conv2D layers.
  - `cnn_dense_activation` (str): Activation function for densely connected classiifer in CNN
  - `optimizer` (str): SGD optimising algorithm. Choose from [`adam`, `rmsprop`, `adagrad`, `sgd`] or other
  keras default strs.
  - `show_summary` (bool): Whether or not to output model.summary()
  - `show_graph` (bool): Whether or not to display a model diagram with keras.utils.plot_model

  OUTPUTS
  - wdnn_model (keras.models.Model): A functional API Wide and Deep Neural Network model with specified hyperparameters.
  """
  num_features = 1034
  weeks_per_consumer = 148
  days_per_week = 7
  channels = 1

  # INPUT LAYERS
  wnn_input = keras.layers.Input(name='wnn_input', shape=(num_features,))
  cnn_input = keras.layers.Input(name='cnn_input', shape=(weeks_per_consumer, 
    days_per_week, channels))

  # WNN HIDDEN LAYERS
  wnn_dense_1 = keras.layers.Dense(activation=wnn_activation, units=wnn_units)(wnn_input)

  # CNN HIDDEN LAYERS
  # Confirming that `n_cnn_filters` is a list before proceeding
  assert type(cnn_filters) == list 
  except AssertionError:
    print("n_cnn_filters is not a list.")
    return

  # Confirming that `n_cnn_filters` has at least one entry
  if (len(cnn_filters) < 1):
    print("n_cnn_filters must contain at least one number.")
    return

  # Proceed only if both functional tests past
  cnn_conv_1 = keras.layers.Conv2D(filters=cnn_filters[0], activation=cnn_activation, 
    padding='same', kernel_size=cnn_kernel_size)(cnn_input)

  # Remember last convolutional layer added
  last_conv_layer = cnn_conv_1 

  # Iterate through remaining layers 
  for i in range(1, n_conv):
    # Add new conv layer that takes previous conv layer's output tensor as input
    new_conv_layer = keras.layers.Conv2D(filters=cnn_filters[i], activation=cnn_activation, 
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