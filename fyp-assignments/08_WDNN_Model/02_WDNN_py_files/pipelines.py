#  IMPORTS
import numpy as np 
import pandas as pd 
import os 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from tensorflow import keras



#  1D PIPELINE
pipeline_1D = Pipeline([
            # Convert all matrices to numpy array for faster processing
            ('to_numpy', FunctionTransformer(pd.DataFrame.to_numpy)),

            # Transpose once so that consumers are now along column axis
            ('row_to_col', FunctionTransformer(np.transpose)), 

            # Use scikit-learn scaler of your choice to scale kWhs on a consumer basis
            ('scaler', StandardScaler()),

            # Retranspose so that consumers are once again along the rows axis
            ('col_to_row', FunctionTransformer(np.transpose)),
])



#  2D PIPELINE
pipeline_2D = Pipeline([
      # Pad each row with zeroes for reshaping 
      ('ZeroPadder', ZeroPadder()),

      # Reshape to a weekly data that can be fed to a CNN
      ('Reshaper_2D', DailyToWeekly())
])