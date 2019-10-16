# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:46:48 2019

@author: lenovo
"""
from collections import Iterable
import statsmodels.api as sm

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from cvxopt import solvers
from cvxopt import matrix

market_engine = create_engine(r'mysql+pymysql://lhxz:postfundlhxz@10.1.101.32:3306/market')
market_conn = market_engine.connect()

def read_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    #data['code'] = data['code'].apply(lambda n:'{:0>6}'.format(n))
    if 'code' in data.columns:
        data = data.set_index(['code', 'date'])
    else:
        data = data.set_index('date')
    return data

def get_data_between_date(data, start_date, end_date):
    index_used = data.index.get_level_values('date')
    return data[(index_used >= start_date) & (index_used < end_date)]

def get_trade_date_before(date, length):
    trade_date_sql = 'select date from trading_calendar where is_trading_day = 1 order by date'
    trade_date = pd.read_sql(trade_date_sql, con=market_conn)
    trade_date = trade_date['date']
    if isinstance(date, Iterable):
        date = [trade_date[trade_date.shift(-length) == d].iloc[0] for d in date]
    else:
        date = trade_date[trade_date.shift(-length) == date].iloc[0]
    return date

def drop_abnormal(hist_data):
    hist_data = hist_data.dropna(how = 'all')
    hist_data = hist_data.dropna(subset = ['close'])
#    hist_data['adj_close'][hist_data['is_ST'] == 1] = np.nan
#    hist_data['adj_close'][hist_data['is_new_stock'] == 1] = np.nan
#    hist_data['adj_close'][hist_data['trade_status'] == 0] = np.nan
#    hist_data['adj_close'][hist_data.returns.abs() > 0.101] = np.nan
    
#    hist_data = hist_data[hist_data['is_ST'] != 1]
#    hist_data = hist_data[hist_data['is_new_stock'] != 1]
#    hist_data = hist_data[hist_data['trade_status'] != 0]
#    hist_data = hist_data[hist_data.returns.abs() <= 0.101]
    return hist_data

def handle_outlier(factors, method = 'median'):
    outlier_function = _handle_outlier_by_median2 if method == 'median' else _handle_outlier_by_std
    factors = pd.DataFrame(factors)
    factors = factors[factors.columns[0]]
    date_group = factors.groupby('date', group_keys = False)
    return date_group.apply(outlier_function)
    #return outlier_function(factors)

def _handle_outlier_by_std(factors_one_day):
    factors_std = factors_one_day.std()
    factors_mean = factors_one_day.mean()
    
    error_observations_index_large = factors_one_day > factors_mean + 5 * factors_std
    error_observations_index_small = factors_one_day < factors_mean - 5 * factors_std
    error_observations_index = error_observations_index_large | error_observations_index_small
    factors_one_day[error_observations_index] = np.nan
    
    outliers_large = factors_one_day > factors_mean + 3 * factors_std
    outliers_small = factors_one_day < factors_mean - 3 * factors_std
    outliers_large = outliers_large & (~ error_observations_index)
    outliers_small = outliers_small & (~ error_observations_index)
    
    factors_one_day[outliers_large] = factors_mean + 3 * factors_std
    factors_one_day[outliers_small] = factors_mean - 3 * factors_std
    
    return factors_one_day

def _handle_outlier_by_median(factors_one_day):
    factors_median = factors_one_day.median()
    deviation_to_median = factors_one_day - factors_median
    median_of_deviation = deviation_to_median.abs().median()
    median_of_deviation *= 1.483
    
    #error_observations_index = deviation_to_median.abs() > 5 * median_of_deviation
    #factors_one_day[error_observations_index] = np.nan
    
    outliers_large = factors_one_day > factors_median + 3 * median_of_deviation
    outliers_small = factors_one_day < factors_median - 3 * median_of_deviation
    isna = factors_one_day.isnull()
    #outliers_large = outliers_large & (~ error_observations_index)
    #outliers_small = outliers_small & (~ error_observations_index)
    
    #max_deviation = deviation_to_median.max()
    #scale = deviation_to_median.abs() / max_deviation
    factors_one_day.loc[outliers_large] = factors_median + 3 * median_of_deviation
    factors_one_day.loc[outliers_small] = factors_median - 3 * median_of_deviation
    factors_one_day.loc[isna] = np.nan
    
    return factors_one_day

def _handle_outlier_by_median2(factors):
    factors_median = factors.groupby('date').transform('median')
    deviation_to_median = factors - factors_median
    median_of_deviation = deviation_to_median.abs().groupby('date').transform('median')
    median_of_deviation *= 1.483
    
    #error_observations_index = deviation_to_median.abs() > 5 * median_of_deviation
    #factors_one_day[error_observations_index] = np.nan
    
    outliers_large = factors > factors_median + 3 * median_of_deviation
    outliers_small = factors < factors_median - 3 * median_of_deviation
    isna = factors.isnull()
    #outliers_large = outliers_large & (~ error_observations_index)
    #outliers_small = outliers_small & (~ error_observations_index)
    
    #max_deviation = deviation_to_median.max()
    #scale = deviation_to_median.abs() / max_deviation
    factors.loc[outliers_large] = factors_median + 3 * median_of_deviation
    factors.loc[outliers_small] = factors_median - 3 * median_of_deviation
    factors.loc[isna] = np.nan
    
    return factors

def fill_missing_values(factors, factor_name, hist_data, fill_value = 'mean'):
    factors = hist_data.join(factors, how = 'left')
    factors_filled = factors.dropna(subset = ['industry'])
    factors_filled = factors_filled.reset_index().set_index(['code', 'date', 'industry'])
    date_industry_group = factors_filled.groupby(['date', 'industry'])
    if fill_value == 'mean':
        indusrty_mean = date_industry_group[factor_name].transform('mean')
        market_mean = factors_filled.groupby('date')[factor_name].transform('mean')
    elif fill_value == 'min':
        indusrty_mean = date_industry_group[factor_name].transform('min')
        market_mean = factors_filled.groupby('date')[factor_name].transform('min')
    elif fill_value == 'q.25':
        indusrty_mean = date_industry_group[factor_name].transform(lambda s:s.quantile(.25))
        market_mean = factors_filled.groupby('date')[factor_name].transform(lambda s:s.quantile(.25))
    elif fill_value == 'q.75':
        indusrty_mean = date_industry_group[factor_name].transform(lambda s:s.quantile(.75))
        market_mean = factors_filled.groupby('date')[factor_name].transform(lambda s:s.quantile(.75))
    elif fill_value == 'max':
        indusrty_mean = date_industry_group[factor_name].transform('max')
        market_mean = factors_filled.groupby('date')[factor_name].transform('max')
    elif fill_value == None:
        pass
    if fill_value:
        is_na = factors_filled[factor_name].isna()
        factors_filled.loc[is_na, factor_name] = indusrty_mean[is_na]
        industry_count = date_industry_group[factor_name].transform(lambda s:len(s))
        factors_filled.loc[is_na & (industry_count < 10), factor_name] = market_mean[is_na & (industry_count < 10)]
    factors_filled = factors_filled.reset_index().set_index(['code', 'date'])
    factors_filled = factors_filled.sort_index()[factor_name]
    return factors_filled
    
def standardlize_factors(factors):
    factor_mean = factors.groupby('date', group_keys = False).transform('mean')
    factor_std = factors.groupby('date', group_keys = False).transform('std')
    return (factors - factor_mean) / factor_std

def preprocess(factors_data, factor_name, hist_data, fill_value = 'mean'):
    factors_data = handle_outlier(factors_data)
    
    factors_data = fill_missing_values(factors_data, factor_name, hist_data, fill_value = fill_value)    
    factors_data = standardlize_factors(factors_data)
    
    return factors_data

def industry_market_value_neutral(factors, hist_data, industry = True, market_value = True):
    if not industry and not market_value:
        raise ValueError("industry与market_value必须至少有一个为True")
    
    name = pd.DataFrame(factors).columns[0]
    hist_data2 = hist_data.copy()
    hist_data2 = hist_data.join(factors, how = 'left')
    neutral_names = []
    if industry:
        neutral_names.append('industry')
    if market_value:
        neutral_names.append('market_value')
    hist_data2 = hist_data2[neutral_names + [name]].dropna()
    def regress_t(data):
        if not market_value:
            industry_dummies = pd.get_dummies(data['industry'])
            X = industry_dummies
        elif not industry:
            X = np.log(data[['market_value']])
        else:
            industry_dummies = pd.get_dummies(data['industry'])
            X = pd.concat([industry_dummies, np.log(data[['market_value']])], axis = 1)
        y = data[name]
        result = sm.OLS(y, X).fit()
        return result.resid
    
    return hist_data2.groupby('date').apply(regress_t)

#def shrinkage_covariance(returns):
    

def mean_variance_optimization(covariance, mean = None, expected_return = None, expected_variance = None, can_short = False):
    if expected_return is not None and expected_variance is not None:
        raise ValueError('不能同时设定均值与方差!')
    
    A = np.ones((covariance.shape[0], 1))
    A = matrix(A.T)
    b = np.ones((1, 1))
    b = matrix(b)
        
    if expected_variance is not None:
        c = -mean.reshape((covariance.shape[0], 1))
        def F(x):
            return x.dot(covariance).dot(x) - expected_variance
        if not can_short:
            G = -np.eye(covariance.shape[0])
            h = np.zeros_like(mean)
        else:
            G, h = None, None
        c = matrix(c)
        G = matrix(G)
        h = matrix(h)
        result = solvers.cpl(c, F, G, h, A, b)
        return np.array(result['x'])
    
    Q = covariance.values
    Q = matrix(Q)
    p = np.zeros((covariance.shape[0], 1))
    p = matrix(p)
    
    if not can_short:
        if expected_return is not None:
            G = np.vstack([-mean.values.reshape((1, covariance.shape[0])), -np.eye(covariance.shape[0])])
            h = np.array([-expected_return].extend([0] * covariance.shape[0]))
        else:
            G = -np.eye(covariance.shape[0])
            h = np.zeros((covariance.shape[0], 1))
    else:
        if expected_return is not None:
            G = -mean.values.reshape((1, covariance.shape[0]))
            h = np.array([-expected_return])
        else:
            G = None
            h = None
    
    G = matrix(G)
    h = matrix(h)
    result = solvers.qp(Q, p, G, h, A, b)
    
    return pd.Series(result['x'], index = covariance.index), result['primal objective']

def portfolio_returns(w, hist_data, rebalance_date, weight_change, transaction_cost = 0):
    hist_data2 = hist_data.copy()
    hist_data2 = hist_data2[hist_data2.index.get_level_values('code').isin(w.columns)]
    hist_data2 = hist_data2['returns'].unstack('code').fillna(0)
    hist_data2 = hist_data2.T.reindex(w.columns).fillna(0).T.sort_index()
    #rebalance_date = w.index
    returns_between_rebalance_date = []
    long_term_returns =  []
    for i in range(len(rebalance_date) - 1):
        start_date, end_date = rebalance_date[i], rebalance_date[i + 1]
        returns = hist_data2[(hist_data2.index > start_date) & (hist_data2.index <= end_date)]
        cum_returns = (returns + 1).cumprod()
        portfolio_cum_returns = cum_returns.dot(w.iloc[i])
        portfolio_cum_returns.iloc[-1] = weight_change.loc[end_date] * portfolio_cum_returns.iloc[-1] / ((1+transaction_cost)/(1-transaction_cost)) + (1 - weight_change.loc[end_date]) * portfolio_cum_returns.iloc[-1]
        long_term_returns_i = portfolio_cum_returns.iloc[-1] - 1
        portfolio_returns = portfolio_cum_returns.pct_change()
        portfolio_returns.iloc[0] = portfolio_cum_returns.iloc[0] - 1
        returns_between_rebalance_date.append(portfolio_returns)
        long_term_returns.append(long_term_returns_i)
    returns = pd.concat(returns_between_rebalance_date)
    long_term_returns = pd.Series(long_term_returns, index = rebalance_date[1:])
    return returns, long_term_returns 

def get_covariance_specific_date(date, hist_data_sub, factors_expo_sub, factors_covariance_dict, specific_variance_sub):
    factors_expo_specific_date = factors_expo_sub[factors_expo_sub.index.get_level_values('date') == date].droplevel('date')
    #industry_expo_specific_date = pd.get_dummies(hist_data_sub['industry'][hist_data_sub.index.get_level_values('date') == date]).droplevel('date')
    #factors_expo_full = factors_expo_specific_date.join(industry_expo_specific_date)
    factors_expo_full = factors_expo_specific_date
    specific_variance_date = specific_variance_sub[specific_variance_sub.index.get_level_values('date') == date].droplevel('date')
    if specific_variance_date.isna().any():
        specific_variance_date = specific_variance_date.dropna()
    specific_covariance_date = pd.DataFrame(np.diag(specific_variance_date), index = specific_variance_date.index, columns = specific_variance_date.index)
    if factors_expo_full.shape[1] < len(factors_covariance_dict[date].columns):
        missing_industry_names = factors_covariance_dict[date].columns.difference(factors_expo_full.columns)
        for c in missing_industry_names:
            factors_expo_full[c] = 0
    factors_expo_full = factors_expo_full[factors_covariance_dict[date].columns]
    covariance = factors_expo_full @ factors_covariance_dict[date] @ factors_expo_full.T
    covariance = covariance.reindex(specific_covariance_date.index)[specific_covariance_date.columns]
    specific_covariance_date = specific_covariance_date.reindex(covariance.index)[covariance.columns]
    covariance += specific_covariance_date
    covariance = covariance.dropna(how = 'all')
    covariance  = covariance.dropna(how = 'all', axis = 1)
    return covariance

def backtest_risk_models(hist_data, factors_expo, factors_covariance_dict, specific_variance, trade_date_list, subset = 'is_hs300'):
    if subset is not None:
        hist_data_sub = hist_data[hist_data[subset] == 1]
    else:
        hist_data_sub = hist_data
    factors_expo_sub = factors_expo.reindex(hist_data_sub.index)
    specific_variance_sub = specific_variance.reindex(hist_data_sub.index)
    weights_dict = {}
    predicted_variance_dict = {}
    for date in trade_date_list[:-1]:
        covariance = get_covariance_specific_date(date, hist_data_sub, factors_expo_sub, factors_covariance_dict, specific_variance_sub)
        weights_on_date, predicted_variance = mean_variance_optimization(covariance)
        weights_dict[date] = weights_on_date
        predicted_variance_dict[date] = predicted_variance
        predicted_variance_dict[date] = weights_on_date @ covariance @ weights_on_date
    weights = pd.DataFrame(weights_dict).fillna(0).T
    daily_returns, long_term_returns = portfolio_returns(weights, hist_data_sub, trade_date_list)
    return daily_returns, long_term_returns, pd.Series(predicted_variance_dict)

def sharpe_ratio(returns):
    sr = returns.mean() / returns.std() * (252 ** 0.5)
    return sr

def find_index_returns(returns):
    index_columns = returns.columns.intersection(['hs300', 'zz500', 'zz800', 'zz1000', 'sz380'])
    index_returns = returns[index_columns[0]]
    portfolio_columns = returns.columns.difference(index_columns)
    excess_returns = returns[portfolio_columns] - np.atleast_2d(index_returns.values).T
    return index_returns, returns[portfolio_columns], excess_returns

def information_ratio(returns):
    index_returns, portfolio_returns, excess_returns = find_index_returns(returns)
    return sharpe_ratio(excess_returns)

def mdd(s):
    s = (s + 1).cumprod()
    max_value = 1
    max_draw_down = 0
    for value in s:
        draw_down = value / max_value - 1
        if draw_down < max_draw_down: max_draw_down = draw_down
        if value > max_value: max_value = value
    return max_draw_down

def tracking_error(returns):
    index_returns, portfolio_returns, excess_returns = find_index_returns(returns)
    return excess_returns.std() * np.sqrt(252)

def longest_positive_length(returns):
    longest_length = 0
    length = 0
    for ret in returns:
        if ret < 0:
            length = 0
        else:
            length+=1
            longest_length = max([length, longest_length])
    return longest_length

def describe_index_enhence(daily_returns, rebalance_returns, turnover, rebalance_time = 20):
    information_dict = {}
    index_returns, portfolio_returns, excess_returns = find_index_returns(daily_returns)
    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns[portfolio_returns.columns[0]]
    if isinstance(excess_returns, pd.DataFrame):
        excess_returns = excess_returns[excess_returns.columns[0]]
    information_dict['annual_returns'] = portfolio_returns.mean() * 252
    information_dict['annual_std'] = portfolio_returns.std() * np.sqrt(252)
    information_dict['max_drawdown'] = mdd(portfolio_returns)
    information_dict['sharpe_ratio'] = information_dict['annual_returns'] / information_dict['annual_std']
    information_dict['annual_excess_returns'] = excess_returns.mean() * 252
    information_dict['tracking_error'] = excess_returns.std() * np.sqrt(252)
    information_dict['information_ratio'] = information_dict['annual_excess_returns'] / information_dict['tracking_error']
    information_dict['excess_max_drawdown'] = mdd(excess_returns)
    information_dict['positive_ratio'] = (rebalance_returns > 0).mean()
    information_dict['positive_sequence_length'] = longest_positive_length(rebalance_returns)
    information_dict['negative_sequence_length'] = longest_positive_length(-rebalance_returns)
    information_dict['daily_excess_win_ratio'] = (excess_returns > 0).mean()
    information_dict['turnover'] = turnover.mean() * 2 * 252 / rebalance_time
    information_dict['win_ratio'] = (rebalance_returns > 0).mean()
    return pd.Series(information_dict)
    