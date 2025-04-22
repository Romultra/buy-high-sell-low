import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Input

def Optimizer_NonProsumer(params, price):
    """ 
    Optimizer function for the non-prosumer model.
    With a single price for buying and selling.

    Parameters:
    params: dictionary with the parameters of the battery.
    price: vector of prices.

    Returns:
    profit: profit value of the optimization problem.
    p_c: vector of charging power values.
    p_d: vector of discharging power values.
    X: vector of battery state of charge values.
    """

    n = len(price)
    
    ### Define the decision variables ###
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    X   = cp.Variable(n)

    ### Define the profit function ###
    profit = cp.sum(-p_c @ price + p_d @ price)
    
    ### Add constraints ###
    constraints = [p_c >= 0, 
                   p_d >= 0, 
                   p_c <= params['Pmax'], 
                   p_d <= params['Pmax']]
    constraints += [X >= params['Cmin'], X <= params['Cmax']]
    constraints += [X[0]==params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']]
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    constraints += [X[n-1]==params['C_n']]
    
    ### Solve the problem ###
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.CLARABEL)
    
    return profit.value, p_c.value, p_d.value, X.value

def profits_from_SOC_strategy(prices, SOC_strategy, negative=True):
    """
    Calculate profits based on the given SOC strategy using the price data from parts of the training set.
    This function takes in the price data, processes it by day, and calculates the profits
    for the battery operation using the Optimizer_NonProsumer function.
    The profit can be returned as a negative value to be minimized.

    Parameters:
    prices (ndarray): The price data for the battery operation.
    SOC_strategy (float): The state of charge strategy for the battery, [0.1, 2].
    negative (bool): If True, the profit will be returned as a negative value.

    Returns:
    float: The total profit calculated for the given SOC strategy.
    """
    # Battery parameters
    battery_params = {
        'Pmax': 1,      # Power capacity in MW
        'Cmax': 2,     # Energy capacity in MWh
        'Cmin': 0.2,      # Minimum SOC (10%)
        'C_0': SOC_strategy,       # Initial SOC
        'C_n': SOC_strategy,       # Final SOC
        'n_c': 0.95,    # Charging efficiency
        'n_d': 0.95     # Discharging efficiency
    }

    # Initialize result
    profits = 0

    # Reshape the prices array to simulate daily data
    daily_prices = prices.reshape(-1, 24)  # 30 days, 24 hours each 

    # Process data by days
    for i, day_prices in enumerate(daily_prices):
        # Start with 50% SOC for the first day, requirement from the assignment
        if i == 0:
            battery_params['C_0'] = 1
        else:
            battery_params['C_0'] = SOC_strategy

        # Optimize battery operation with price arbitrage
        # The goal is to charge when prices are low and discharge when prices are high
        profit_value, p_c_value, p_d_value, X_value = Optimizer_NonProsumer(battery_params, day_prices)

        # Calculate the battery net discharge.
        # Positive means discharging/selling, negative means charging/buying
        net_discharge = p_d_value - p_c_value

        # Calculate cost with battery
        day_profit = 0
        for j in range(len(net_discharge)):
            day_profit += net_discharge[j] * day_prices[j]

        profits += day_profit

    # If negative is True, return the negative profit for minimization
    if negative:
        return -profits
    else:
        return profits

def create_sequences(data, window_size, n_lookahead):
    X, y = [], []
    for i in range(len(data) - window_size - n_lookahead + 1):
        X.append(data[i:i+window_size])  # Input sequence of length <window_size>
        y.append(data[i+window_size:i+window_size+n_lookahead])  # Output: next <n_lookahead> prices
    return np.array(X), np.array(y)

def create_sequences_multivariate(data, window_size, n_lookahead):
    X, y = [], []
    for i in range(len(data) - window_size - n_lookahead +1):
        X.append(data[i:i+window_size])              # Input: <window_size> x <n_features>, ie 24x4
        y.append(data[i+window_size:i+window_size+n_lookahead, 0])  # Output: next <n_lookahead> prices
    return np.array(X), np.array(y)

# Rolling forecast
def rolling_forecast(model, history_scaled, test_scaled, window_size, n_lookahead):
    predictions = []
    data = np.concatenate((history_scaled, test_scaled), axis=0)  # Concatenate history and test data

    # Number of predictions
    n_train = len(history_scaled)  # Training set size
    n_test = len(test_scaled) # Test set size
    N = int(n_test/n_lookahead)
    
    for day in range(N):
        input_seq = data[n_train+day*n_lookahead-window_size:n_train+day*n_lookahead].reshape((1, window_size, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0]

        predictions.append(pred_scaled)

    return np.array(predictions)

def rolling_forecast_multivariate(model, history_scaled, test_scaled, n_steps, n_lookahead, n_features): 
    predictions = []
    data = np.concatenate((history_scaled, test_scaled), axis=0)  # Concatenate history and test data
    
    # Number of predictions
    n_train = len(history_scaled)  # Training set size
    n_test = len(test_scaled) # Test set size
    N = int(n_test/n_lookahead)

    for day in range(N):
        input_seq = data[n_train+day*n_lookahead-n_steps:n_train+day*n_lookahead].reshape((1, n_steps, n_features))
        pred_scaled = model.predict(input_seq, verbose=0)[0]
                
        predictions.append(pred_scaled)

    return np.array(predictions)

def LSTM_1layer(n_steps, n_features, n_neurons, n_lookahead, dropout):
    
    # Define model
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))
    model.add(LSTM(n_neurons, dropout=dropout))
    model.add(Dense(n_lookahead))
    model.compile(loss='mse', optimizer='adam')

    return model

def LSTM_multilayer(n_steps, n_features, n_neurons, 
                    n_neurons_dense, n_lookahead, dropout1, dropout2):
    
    # Define model
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))
    model.add(LSTM(n_neurons, activation='relu', return_sequences=True))
    model.add(Dropout(dropout1))  # Dropout layer
    model.add(LSTM(n_neurons, activation='relu'))
    model.add(Dropout(dropout2))  # Dropout layer
    model.add(Dense(n_neurons_dense, activation='relu'))
    model.add(Dense(n_lookahead))
    model.compile(optimizer='adam', loss='mse')
    
    return model

def FitLSTM_multilayer(train_scaled, n_steps, n_features, n_lookahead, n_neurons, 
            n_neurons_dense, epochs, dropout1, dropout2):
         
    # Split the training data into input-output pairs using a sliding window approach
    # X will contain the sequences and y will contain the corresponding targets
    X, y = create_sequences(train_scaled, n_steps, n_lookahead)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape X to be 3D for LSTM input

    # Create a multilayer LSTM model with the specified parameters
    # The model will be built using n_steps, n_features, and other hyperparameters
    model = LSTM_multilayer(n_steps, n_features, n_neurons, 
                        n_neurons_dense, n_lookahead, dropout1, dropout2)

    # Fit the model on the training data using a validation split of 20%
    # The model will train on 80% of the training data and validate on 20% for each epoch
    history = model.fit(
    X, y, 
    epochs=epochs, 
    validation_split=0.2,  # 20% of training data used for validation
    verbose=1)  # Display progress during training

    # Display the model architecture summary
    model.summary()

    # Plot the training and validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')  # Training loss over epochs
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Validation loss over epochs
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  # Display the legend at the upper right
    plt.grid(True)  # Add a grid for easier visualization of the loss values
    plt.show()

    # Return the trained model and the scaled dataset
    return model

def FitLSTM_1layer(train_scaled, n_steps, n_features, n_lookahead, n_neurons, 
                   epochs, dropout, ordered_validation=True):
         
    # Split the training data into input-output pairs using a sliding window approach
    # X will contain the sequences and y will contain the corresponding targets
    X, y = create_sequences(train_scaled, n_steps, n_lookahead)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape X to be 3D for LSTM input

    # Create a 1 layer LSTM model with the specified parameters
    # The model will be built using n_steps, n_features, and other hyperparameters
    model = LSTM_1layer(n_steps, n_features, n_neurons, n_lookahead, dropout)

    # Fit the model on the training data using a validation split of 20%
    # The model will train on 80% of the training data and validate on 20% for each epoch
    if ordered_validation:
        # Split the data into training and validation sets in order
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]   

        history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, verbose=1)
    else:
        # Randomly split the data into training and validation sets
        history = model.fit(
        X, y, epochs=epochs, 
        validation_split=0.2,  # 20% of training data used for validation
        verbose=1)  # Display progress during training

    # Display the model architecture summary
    model.summary()
    
    # Plot the training and validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')  # Training loss over epochs
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Validation loss over epochs
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  # Display the legend at the upper right
    plt.grid(True)  # Add a grid for easier visualization of the loss values
    plt.show()

    # Return the trained model and the scaled dataset
    return model

def FitLSTM_ext(train_scaled, n_steps, n_features, n_lookahead, n_neurons, 
                epochs, dropout, ordered_validation=True):

    # Split the training data into input-output pairs using a sliding window approach
    # X will contain the sequences and y will contain the corresponding targets
    X, y = create_sequences_multivariate(train_scaled, n_steps, n_lookahead)

    # Create a 1 layer LSTM model with the specified parameters
    # The model will be built using n_steps, n_features, and other hyperparameters
    model = LSTM_1layer(n_steps, n_features, n_neurons, n_lookahead, dropout)

    # Fit the model on the training data using a validation split of 10%
    # The model will train on 80% of the training data and validate on 10% for each epoch
    if ordered_validation:
        # Split the data into training and validation sets in order
        split_index_start = int(len(X) * 0.4)
        split_index_end = int(len(X) * 0.5)
        X_train, X_val = np.concatenate((X[:split_index_start], X[split_index_end:])), X[split_index_start:split_index_end]
        y_train, y_val = np.concatenate((y[:split_index_start], y[split_index_end:])), y[split_index_start:split_index_end]

        history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, verbose=1)
    else:
        # Randomly split the data into training and validation sets
        history = model.fit(
        X, y, epochs=epochs, 
        validation_split=0.1,  # 10% of training data used for validation
        verbose=1)  # Display progress during training

    # Display the model architecture summary
    model.summary()

    # Plot the training and validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')  # Training loss over epochs
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Validation loss over epochs
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  # Display the legend at the upper right
    plt.grid(True)  # Add a grid for easier visualization of the loss values
    plt.show()

    # Return the trained model and the scaled dataset
    return model