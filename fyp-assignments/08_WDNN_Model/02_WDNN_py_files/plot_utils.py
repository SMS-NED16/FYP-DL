#  IMPORTs 
# For generating plots 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')

# For saving and loading files with Python 
import pickle





#  PLOT HISTORY 
def plot_history(history, model_name=None, figsize=(12, 6), 
                 loss_name='Binary Crossentropy'):
  # Extract loss from the history object
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)

  # Extract AUCs
  auc = None
  val_auc = None

  # Extract `auc` and `val_auc` regardless of last number in key
  for key in list(history.history.keys()):
    if 'auc' in key and 'val' not in key:
      auc = history.history[key]
    elif 'auc'in key and 'val' in key:
      val_auc = history.history[key]

  # Plotting Loss
  plt.figure(figsize=figsize)
  plt.plot(epochs, loss, label='Training Loss')
  plt.plot(epochs, val_loss, label='Validation Loss')
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel(loss_name, fontsize=14)
  plt.title('Loss ' + ('' if model_name is None else ' - {}'.format(model_name)), 
            fontsize=18)
  plt.legend()

  # Plotting AUC - only if in dictionary
  if auc is not None and val_auc is not None:
    plt.figure(figsize=figsize)
    plt.plot(epochs, auc, label='Training AUC')
    plt.plot(epochs, val_auc, label='Validation AUC')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    plt.title('AUC ' + ('' if model_name is None else' - {}'.format(model_name)), 
              fontsize=18)
    plt.legend()



#  PLOT TRAINING TRENDS
def plot_training_trends(history_list, labels_list, suptitle):
  # Empty lists first
  lost_lists = []
  val_loss_lists = []
  auc_lists = []
  val_auc_lists = []

  # Extracting list of training losses 
  loss_lists = [history['loss'] for history in history_list]
  val_loss_lists = [history['val_loss'] for history in history_list]

  # Epochs will be the same as the length of any given loss list
  epochs = range(1, len(loss_lists[0]) + 1)

  # Extracting list of accuracies will be more difficult - uses functions defined earlier
  auc_lists = get_auc(history_list)
  val_auc_lists = get_val_auc(history_list)

  # Labels for each plot in same order as dropout rates of models
  legend_labels = labels_list

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20, 10))

  for i in range(0, len(loss_lists)):
    ax1.plot(epochs, loss_lists[i])
    ax2.plot(epochs, val_loss_lists[i])
    ax3.plot(epochs, auc_lists[i])
    ax4.plot(epochs, val_auc_lists[i])

  for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Epochs')
    ax.legend(legend_labels) 

  ax1.set_title('Training Loss'); ax2.set_title('Validation Loss')
  ax3.set_title('Training AUC'); ax4.set_title('Validation AUC')
  fig.suptitle(suptitle, fontsize=18)



#  GET AUC
def get_auc(histories):
  auc_list = []
  for history in histories:
    for key in history.keys():
      if 'auc' in key and 'val' not in key:
         auc_list.append(history[key])
  
  return auc_list



#  GET VAL AUC
def get_val_auc(histories):
  val_auc_list = []
  for history in histories:
    for key in history.keys():
      if 'auc' in key and 'val' in key:
        val_auc_list.append(history[key])
        
  return val_auc_list



#  EXTRACR HISTORY
def extract_histories(history_dict):
  """Every `history` object returned by a `keras` `fit` method is actually weird.
  The actual history is `history.history`. Extracting this, while maintaining the
  same key. Idk. I am too tired to write a good docstring."""

  # Dictionary of the history objects we want
  extracted_histories = {}

  # Index into the history object passed from model.fit, and extract the actual `history` 
  for (history_name, history_obj) in history_dict.items():
    extracted_histories[history_name] = history_obj.history
  
  # Return the extracted history
  return extracted_histories



#  SAVE TRAINING HISTORY
def save_training_history(histories, file_name='test_histories', extension='pickle', download=False):
  """Saves dictionary of history dictionaries to a pickle file that can be loaded later"""
  # Convert histories in history.history objects 
  history_objs = extract_histories(histories)
  
  # Create a file name of the right extension - I recommend pickle 
  save_file_name = file_name + '.' + extension 

  # Save this pickle file to disk
  with open(save_file_name, 'wb') as pkl_file:
    pickle.dump(history_objs, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

  # If also requested to download automatically
  if download:
    files.download(save_file_name)
  


#  LOAD TRAINING HISTORY
def load_training_history(file_name):
  """Loads a dictionary of history dictionaries from file"""
  with open(file_name, 'rb') as pkl_file:
      return pickle.load(pkl_file)

