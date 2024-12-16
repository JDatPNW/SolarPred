from flask import Flask, render_template
import pandas as pd
import numpy as np
from datetime import datetime as dt
import tensorflow as tf
from sqlalchemy import create_engine


DATABASE_URI = 'YOUR_DATABASE_URI'

engine = create_engine(DATABASE_URI)

def load_data():
    try:
        # Write your SQL query
        sql_query = "SELECT * FROM dataset"

        # Execute the query and fetch data into a pandas DataFrame
        df_data = pd.read_sql(sql_query, engine)
        df_data = df_data.drop("id", axis=1)
        df_data.set_index('date', inplace=True)
        df_data.index = pd.to_datetime(df_data.index)

        # Return the dataframe as JSON (you can also return as HTML, etc.)
        return df_data

    except Exception as e:
        return str(e)


def df_to_X_y(df, window_size=5):  # 5 here means we are taking the last 5 values
    df_as_np = df.to_numpy() #converting the dataframe to numpy arrays
    X = []
    y = []
    t = []
    time = df.index.values

    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size, 0]
        y.append(label)
        t.append(time[i+window_size])
    return np.array(X), np.array(y), np.array(t)

def prep_data(df, WINDOW_SIZE):
    X, y, t = df_to_X_y(df, WINDOW_SIZE)
    X = np.squeeze(X)
    y = np.squeeze(y)
    X_train, y_train  = X[:8000], y[:8000]
    X_val, y_val  = X[8000:8400], y[8000:8400]
    X_test, y_test  = X[8400:], y[8400:]
    t_test = t[8400:]
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    return X_train, X_val, X_test, y_train, y_val, y_test, t_test



def load_model():
    model = tf.keras.saving.load_model('./ai/model.keras')  # Use the correct file extension and path
    return model





# Autoregression function for inference
def autoregressive_predict(model, initial_input, n_steps):
    """
    Generates autoregressive predictions for `n_steps` future time steps.

    Parameters:
    - model: Trained model to use for predictions
    - initial_input: Initial input array with shape (1, time_steps, features)
    - n_steps: Number of time steps to predict autoregressively

    Returns:
    - predictions: Array of predicted values for `n_steps` time steps
    """
    # Ensure the input has batch dimension
    current_input = np.expand_dims(initial_input, axis=0)
    predictions = []

    for _ in range(n_steps):
        # Make a single-step prediction
        next_step_pred = model.predict(current_input, verbose=0)
        predictions.append(next_step_pred[0][0])  # Collect the prediction

        # Use the prediction as the new input (shift input window)
        # Remove the oldest time step and add the new prediction at the end
        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, 0] = next_step_pred[0][0]  # Update last step with prediction

    return np.array(predictions)


def reverse_normalization(normalized_value, col_min, col_max):
    return round(normalized_value * (col_max - col_min) + col_min, 2)

def get_most_recent_year_for_each_month(df):
    # Create an empty list to hold the filtered data for each month
    filtered_data = []
    
    # Iterate through each month (1 to 12)
    for month in range(1, 13):
        # Get the most recent year for this month
        most_recent_year = df[df.index.month == month].index.year.max()
        
        # Filter the data for the most recent year and the current month
        month_data = df[(df.index.month == month) & (df.index.year == most_recent_year)]
        
        # Append the filtered data to the list
        filtered_data.append(month_data)
    
    # Concatenate the filtered data for each month
    return pd.concat(filtered_data)


def get_monthly(df_data):
    # Resample by month and sum the 'generation' column
    # Get the most recent year from the data
    df_filtered = get_most_recent_year_for_each_month(df_data)

    monthly_generation = df_filtered['generation'].resample('M').sum()

    # Get the last 8 months
    last_8_months = monthly_generation[-13:-1]
    return last_8_months



app = Flask(__name__)

@app.route('/')
def dashboard():

    # Example usage
    # Start with a sample from the test data, e.g., the first test sequence
    WINDOW_SIZE = 24

    min_gen = 0.0
    max_gen = 389712.0

    data = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, t_test = prep_data(data, WINDOW_SIZE)
    model = load_model()
    
    start_step = 202  # Start from the 10th data point in X_test
    initial_input = X_test[start_step]  # Shape: (24, 9)
    current = y_test[start_step-WINDOW_SIZE:start_step]  # Shape: (24, 9)
    time = t_test[start_step-WINDOW_SIZE:start_step]
    n_steps = 24  # Number of steps to autoregressively predict
    auto_reg_model = model

    predictions = autoregressive_predict(auto_reg_model, initial_input, n_steps)

    current_generation_list = current.tolist()  # example current value
    predicted_generation_list = predictions.tolist()  # example predicted value

    real_predicted_generation_list = [reverse_normalization(v, min_gen, max_gen) for v in predicted_generation_list]
    real_current_generation_list = [reverse_normalization(v, min_gen, max_gen) for v in current_generation_list]

    current_generation = real_current_generation_list[-1]  # example current value
    predicted_generation = real_predicted_generation_list[0]  # example predicted value


    formatted_time = [str(t).replace('T', ' ').split('.')[0] for t in time]

    # Further format the string to match the desired pattern: year.month.day h:tm
    # Append the new time n times (e.g., 5 times in this case)
    # Convert the last datetime string to numpy.datetime64
    last_time = np.datetime64(formatted_time[-1], 'm')

    # Add one hour to the last time
    new_time = last_time + np.timedelta64(1, 'h')

    # Convert back to string in desired format
    new_time_str = str(new_time)[:16].replace('T', ' ')

    n = n_steps
    for _ in range(n):
        formatted_time.append(f"{new_time_str[:4]}.{new_time_str[5:7]}.{new_time_str[8:10]} {new_time_str[11:13]}:{new_time_str[14:16]}")
        # Increment the time for next iteration
        new_time += np.timedelta64(1, 'h')
        new_time_str = str(new_time)[:16].replace('T', ' ')


    # Generate autoregressive predictions

    real_predicted_generation_list.insert(0, real_current_generation_list[-1])

    for i in range(WINDOW_SIZE-1):
        real_predicted_generation_list.insert(0, 'NaN')

    months = get_monthly(data)
    month_nums = months.values.tolist()
    month_nums = [reverse_normalization(v, min_gen, max_gen) for v in month_nums]
    month_nums = [round(float(i), 2) for i in month_nums]

    # Get the month names as a list
    month_names = months.index.strftime('%B %Y').tolist()

  

    return render_template('dashboard.html', current_generation_list=real_current_generation_list, predicted_generation_list=real_predicted_generation_list, 
                           current_generation=current_generation , predicted_generation=predicted_generation, formatted_time=formatted_time,
                           month_nums=month_nums, month_names=month_names)

if __name__ == '__main__':
    app.run(debug=True)

