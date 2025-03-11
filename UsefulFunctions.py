import os
import pandas as pd
import cvxpy as cp

def PricesDK(df_prices):
    
    ### Calculate the fixed Tax column ###
    df_prices["Tax"] = 0.8
    
    ### Calculate the fixed TSO column ###
    
    df_prices["TSO"] = 0.1
    
    ### Add the DSO tariffs ###
    
    ### The Low period has the same price during both summer/winter periods ###
    df_prices.loc[df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5]),
                  "DSO"] = 0.15
    
    ### Peak period in winter ###
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([1,2,3,10,11,12]))
                  & (df_prices["HourDK"].dt.hour.isin([17,18,19,20])),
                  "DSO"] = 1.35
    
    ### Peak period in summer ###
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9]))
                  & (df_prices["HourDK"].dt.hour.isin([17,18,19,20])),
                  "DSO"] = 0.6
    
    ### High period in winter ###
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([1,2,3,10,11,12]))
                  & (df_prices["HourDK"].dt.hour.isin([6,7,8,9,10,11,12,13,14,15,16,21,22,23])),
                  "DSO"] = 0.45
    
    ### High period in summer ###
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9]))
                  & (df_prices["HourDK"].dt.hour.isin([6,7,8,9,10,11,12,13,14,15,16,21,22,23])),
                  "DSO"] = 0.23
    
    ### Calculate VAT ###
    df_prices["VAT"] = 0.25*(df_prices["Tax"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["Sell"])
    
    ### Calculate Buy price ###
    df_prices["Buy"] = df_prices["Tax"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["Sell"]+df_prices["VAT"]
    
    return df_prices


def LoadData():
    
    ### Load electricity prices ###
    price_path = os.path.join(os.getcwd(),'ElspotpricesEA.csv')
    df_prices = pd.read_csv(price_path)
    
    ### Convert to datetime ###
    df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])
    df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
    df_prices['HourUTC'] = df_prices['HourUTC'].dt.tz_localize('UTC')
    df_prices['HourDK'] = df_prices['HourUTC'].dt.tz_convert('CET')
    
    ### Convert prices to DKK/kWh ###
    df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK']/1000
    
    ### Filter only DK2 prices ###
    df_prices = df_prices.loc[df_prices['PriceArea']=="DK2"]
    
    ### Keep the time and price columns ###
    df_prices = df_prices[['HourDK','SpotPriceDKK',"HourUTC"]]
    
    ### Reset the index ###
    df_prices = df_prices.reset_index(drop=True)

    ### Keep only 2022 and 2023 ###
    df_prices = df_prices.loc[df_prices["HourDK"].dt.year.isin([2022,2023])]
    
    ### Reset the index ###
    df_prices = df_prices.reset_index(drop=True)
    
    ### Rename SpotPriceDKK to Sell ###
    df_prices.rename(columns={'SpotPriceDKK': 'Sell'}, inplace=True)
    
    ###  Load prosumer data ###
    file_P = os.path.join(os.getcwd(),'ProsumerHourly.csv')
    df_pro = pd.read_csv(file_P)
    df_pro["TimeDK"] = pd.to_datetime(df_pro["TimeDK"])
    df_pro['TimeDK'] = pd.to_datetime(df_pro['TimeDK']).dt.tz_localize('CET', ambiguous='infer')
    df_pro["TimeUTC"] = pd.to_datetime(df_pro["TimeUTC"])
    df_pro['TimeUTC'] = df_pro['TimeUTC'].dt.tz_localize('UTC')
    df_pro.rename(columns={'Consumption': 'Load'}, inplace=True)
    df_pro.rename(columns={'TimeDK': 'HourDK'}, inplace=True)
    df_pro.rename(columns={'TimeUTC': 'HourUTC'}, inplace=True)
    df_pro = df_pro.reset_index(drop=True)

    return df_prices, df_pro

def Optimizer(params, ps, pb, net_load):

    """ 
    Calculate the dimension of your decision variables (n)
    # Do not hard-code values (i.e. n = 24!)
    # A day may have 23 or 25 hours or you may want to solve your problem over 48 hours!
    """
    
    n = len(ps)
    
    ### Define the decision variables ###
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    X   = cp.Variable(n)

    ### Define the profit variable ###
    f_plus = cp.Variable(n, nonneg=True)
    f_minus = cp.Variable(n, nonneg=True)
    
    ### Define the flow variable ###
    flow = net_load + p_c - p_d

    ### Define the profit function ###
    profit = cp.sum( f_minus @ ps - f_plus @ pb )
    
    ### Add constraints ###
    constraints = [p_c >= 0, 
                   p_d >= 0, 
                   p_c <= params['Pmax'], 
                   p_d <= params['Pmax']]
    ### Add the flow constraints ###
    constraints += [f_plus - f_minus == flow,
                    f_plus >= 0,
                    f_minus >= 0]
    constraints += [X >= 0, X <= params['Cmax']]
    constraints += [X[0]==params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']]
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    constraints += [X[n-1]>=params['C_n']]
    
    ### Solve the problem ###
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.CLARABEL)
    
    return profit.value, p_c.value, p_d.value, X.value