import cvxpy as cp
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from keras.optimizers import Adam # type: ignore

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
    constraints += [X[n-1]>=params['C_n']]
    
    ### Solve the problem ###
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.CLARABEL)
    
    return profit.value, p_c.value, p_d.value, X.value


def LSTM_1layer(n_steps, n_features, X_train, y_train, n_neurons, n_lookahead, dropout):
    
    # define model
    # n_steps=24
    # n_features=1
    # n_neurons=50
    # dropout=0.05

    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))
    model.add(LSTM(n_neurons, activation='relu', dropout=dropout))
    model.add(Dense(n_lookahead))
    # Compile model with gradient clipping
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def LSTM_multilayer(n_steps, n_features, X_train, y_train, n_neurons, 
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