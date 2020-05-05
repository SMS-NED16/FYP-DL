# Imports for 1D pipeline functions
from pandas import DataFrame
from numpy import transpose 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, FunctionTransformer

# Imports for 2D pipeline functions
from wdnn_transformers import ZeroPadder, DailyToWeekly

def get_feature_scaler(scaling_strategy='Standard'):
      """Helper function to return a Scikit-Learn Scaler object for use with a 
      Pipeline based on a `scaling_strategy`. 

      INPUTS
      - `scaling_strategy`(str) one of ['Standard', 'MinMax', 'MaxAbs', 'Robust'] corresponding
      to Scikit-Learn StandardScaler, MinMaxScaler, MaxAbsScaler, and RobustScaler respectively.

      OUTPUTS
      - Scikit-Learn `Scaler` object
      """

      # If user has specified a valid scaling strategy string
      if scaling_strategy in ['Standard', 'MinMax', 'MaxAbs', 'Robust']:
            # Use scaling strategy string to index dictionary and get correct scaler
            return {
                  'Standard': StandardScaler(),
                  'MinMax': MinMaxScaler(),
                  'MaxAbs': MaxAbsScaler(), 
                  'Robust': RobustScaler(),
            }.get(scaling_strategy)
      else:
            # Return StandardScaler if user's specified scaling strategy str is invalid
            print(f"{scaling_strategy} is an invalid scaling strategy. Using default.")
            return StandardScaler()

def get_pipeline_1D(scaling_strategy):
      """Returns a Scikit-Learn Pipeline object that preprocesses 1D kWh data by 
      apply feature scaling on a consumer basis. 
      
      INPUTS
      - `scaling_strategy`(str):- `scaling_strategy`(str) one of ['Standard', 'MinMax', 
      'MaxAbs', 'Robust'] corresponding to Scikit-Learn StandardScaler, MinMaxScaler,
       MaxAbsScaler, and RobustScaler respectively.

      OUTPUTS
      - Scikit-Learn `Pipeline` object that converts data to numpy array, transposes it,
      scales it according to specified scaling strategy, retransposes it.""" 

      # Instantiate a Pipeline object with the right scaling strategy and all other steps
      pipeline_1D = Pipeline([
          # Convert all dfs to numpy array for faster processing
          ('to_numpy', FunctionTransformer(DataFrame.to_numpy)),

          # Transpose once so that consumers are now along column axis
          ('row_to_col', FunctionTransformer(transpose)), 

          # Use scikit-learn scaler of your choice to scale kWhs on a consumer basis
          ('scaler', get_feature_scaler(scaling_strategy)),

          # Retranspose so that consumers are once again along the rows axis
          ('col_to_row', FunctionTransformer(transpose)),
      ])

      # Return the pipeline 
      return pipeline_1D

def get_pipeline_2D():
      """Returns a Scikit-Learn Pipeline object for processing 1D daily kWh data into
      2D weekly kWh data for use with a CNN. Must be used with the output of a 
      pipeline_1D object, or (m, n)-dimensional NumPy matrix
      
      INPUTS
      - None

      OUTPUTS
      - A Scikit-Learn `Pipeline` object that will apply `ZeroPadder` and 
      `Reshaper_2D` transformations to input data in that order."""

      # Instantiate a Pipeline object with custom transformer classes 
      pipeline_2D = Pipeline([
            # Pad each row with zeroes for reshaping 
            ('ZeroPadder', ZeroPadder()),

            # Reshape to a weekly data that can be fed to a CNN
            ('Reshaper_2D', DailyToWeekly())
      ])

      # Return the 2D pipeline 
      return pipeline_2D
