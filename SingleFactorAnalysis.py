# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:45:03 2019

@author: lenovo
"""
import time
import pickle

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Day

import utils

class SFPortfolio:
    
    def __init__(self, factors, hist_data, only_rebalance_date = False, 
                 subset = 'all', preprocess = True, market_value_neutral = True, 
                 industry_neutral = True, fill_value = 'mean', group_num = 5, 
                 balance_time = '1M', weights = 'EW', select_from_industry = False):
        '''
        factors: 要研究的因子数据
        hist_data: 股票历史价量数据
        only_rebalance_date: 因子数据是否只包含调仓日数据
        subset：从某个行业或者指数选股
        preprocess：是否对因子进行去极值、缺失值填充、标准化处理
        market_value_neutral：是否对因子进行市值中性处理
        industry_neutral:是否对因子做行业中性处理
        fill_value：填补缺失值的方式，行业均值/行业中位数/行业最大最小值
        group_num: 要把股票分为几组
        balance_time: 调仓周期
        weights: 股票权重序列，EW代表等权
        select_from_industry:是否将股票按照行业分组
        '''
        self._factors = factors.copy()
        self._balance_time = balance_time
        self._weights = weights
        self._subset = subset
        self._hist_data = hist_data.copy()
        self._group_num = group_num
        self._market_neutral = market_value_neutral
        self._industry_neutral = industry_neutral
        self._fill_value = fill_value
        self._preprocess = preprocess
        self._only_rebalance_date = only_rebalance_date
        self._select_from_industry = select_from_industry
        
        self._IC_data = None
        self._rebalance_date = self._cal_rebalance_date()
        self._group = None
        self._returns = None
        self._sub_data = self._extract_rebalance_day_data()
    
    def _extract_rebalance_day_data(self):
        '''
        提取填仓日的因子数据和股票价格数据，并根据参数进行因子数据预处理
        '''
        sub_date =  self._rebalance_date
        hist_data_sub = self._hist_data[self._hist_data.index.get_level_values('date').isin(sub_date)]
        
        factors_sub  = self._factors[self._factors.index.get_level_values('date').isin(sub_date)]
        factor_name = pd.DataFrame(factors_sub).columns[0]
        if self._preprocess:#因子数据预处理
            factors_sub = utils.preprocess(factors_sub, factor_name, hist_data_sub, fill_value = self._fill_value)
        if self._market_neutral or self._industry_neutral:#行业市值中性
            factors_sub = utils.industry_market_value_neutral(factors_sub, hist_data_sub, industry = self._industry_neutral, market_value = self._market_neutral)
            factors_sub = pd.Series(factors_sub)
            factors_sub = factors_sub.droplevel(level = 0)
            factors_sub.name = factor_name
        
        hist_data_sub = hist_data_sub.sort_index()
        hist_data_sub['period_returns'] = hist_data_sub['adj_close'].groupby('code').apply(lambda s:s.shift(-1)/s - 1)
        hist_data_sub = hist_data_sub[(hist_data_sub['is_ST'] == 0) & (hist_data_sub['is_new_stock'] == 0) & (hist_data_sub['status'] == 1)]
        hist_data_sub = hist_data_sub.join(pd.DataFrame(factors_sub).rename(columns = {factor_name:'factors'}))
        if self._subset != 'all':
            hist_data_sub = hist_data_sub[hist_data_sub[self._subset] == 1]
        return hist_data_sub.dropna(subset = ['factors'])
    
    def _cal_rebalance_date(self):
        '''
        计算调仓日期
        '''
        if self._only_rebalance_date:#如果因子值只包含调仓日数据则提取因子序列的日期index
            trade_date = self._factors.index.get_level_values('date').unique().sort_values()
            return trade_date
        #否则从历史数据提取日期序列
        unique_date = self._hist_data.index.get_level_values('date').unique().sort_values()
        trade_date = list(unique_date[::self._balance_time])
        last_date = list(unique_date)[-1]
        if last_date != trade_date[-1]:
            trade_date += [last_date]
        return pd.Series(trade_date)
    
    def _cal_portfolio_returns_between_balancing(self):
        '''
        计算股票组合股票组合日度收益率，速度慢
        '''
        print('_cal_portfolio_returns_between_balancing--1',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        if self._weights == 'MV':
            self._hist_data['weights'] = self._hist_data['market_value']
            
        used_columns = [] if self._weights == 'EW' else ['weights']
        hist_data = pd.concat([self._hist_data[['returns'] + used_columns], self._group], axis = 1)
        gross_returns = hist_data
            
        returns_date = gross_returns.index.get_level_values('date')
        portfolio_returns_between_balancing = [0] * (len(self._rebalance_date) - 1)
        
        for i in range(len(self._rebalance_date) - 1):
            #The start and end of a period between balancing
            start_date, end_date = self._rebalance_date[i], self._rebalance_date[i + 1]
            if i == len(self._rebalance_date) - 2: end_date += Day(1)
            #history data during the period
            returns_between_balancing = gross_returns[(returns_date >= start_date) & (returns_date < end_date)]
            returns_between_balancing = (returns_between_balancing['returns'].fillna(0) + 1).groupby('code', group_keys = False).cumprod()
            portfolio_returns_between_balancing[i] = returns_between_balancing
        
        print('_cal_portfolio_returns_between_balancing--2', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        cum_returns_stocks = pd.concat(portfolio_returns_between_balancing).sort_index()
        cum_returns_stocks.name = 'cum_returns'
        cum_returns_stocks = pd.concat([cum_returns_stocks, gross_returns[['group'] + used_columns]], axis = 1)
        #Calculate the portfolio value if start from 1
        group_data = cum_returns_stocks[['cum_returns', 'group'] + used_columns].groupby(['date', 'group'])
        if self._weights == 'EW':
            cum_returns = group_data.mean()
        else:
            cum_returns =group_data.apply(lambda df:np.average(df.cum_returns, weights = df.weights)) 
        
        print('_cal_portfolio_returns_between_balancing--3', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        cum_returns = cum_returns.unstack(level = 'group')
        returns_date = cum_returns.index = cum_returns.index.get_level_values('date')
        for i in range(len(self._rebalance_date) - 1):
            #The start and end of a period between balancing
            start_date, end_date = self._rebalance_date[i], self._rebalance_date[i + 1]
            if i == len(self._rebalance_date) - 2: end_date += Day(1)
            
            cum_returns_between_balancing = cum_returns[(returns_date >= start_date) & (returns_date < end_date)]
            returns_between_balancing = cum_returns_between_balancing.pct_change()
            if len(cum_returns_between_balancing) != 0:
                returns_between_balancing.iloc[0] = cum_returns_between_balancing.iloc[0] - 1
            
            portfolio_returns_between_balancing[i] = returns_between_balancing
        
        print('_cal_portfolio_returns_between_balancing--4', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        return pd.concat(portfolio_returns_between_balancing).sort_index()
    
    
    def _rank_and_divide(self):
        '''
        将因子排序并分组，每一组组成一个资产组合
        '''
        print('_rank_and_divide--1', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        factors_rank = self._sub_data['factors']
        
        if self._select_from_industry:#按照行业排序并分组
            factors_rank = self._sub_data[['factors', 'industry']]
            factors_rank = factors_rank.set_index('industry', append = True)
            factors_rank = factors_rank.groupby(['date', 'industry'], group_keys = False).rank(pct = True)
            factors_rank = factors_rank.droplevel(level = 'industry')
        else:#全市场排序分组
            factors_rank = factors_rank.groupby('date', group_keys = False).rank(pct = True)
        #else:
        #    industry = self._hist_data['industry'][self._hist_data.index.get_level_values(level = 'date').isin(self._rebalance_date)]
        #    factors_rank = pd.concat([factors_rank, industry], axis = 1)
        #    factors_rank = factors_rank.groupby(['date', 'industry'], group_keys = False).rank(pct = True)
        #    #del factors_rank['industry']
            
        factors_rank = pd.DataFrame(factors_rank).sort_index()
	
        factors_rank.columns = ['group']
        percent_group = 100 // self._group_num
        factors_rank = factors_rank * 100 // percent_group + 1
        factors_rank[factors_rank == self._group_num + 1] = self._group_num
        
        self._group = pd.DataFrame(factors_rank)
        print('_rank_and_divide--2', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
	
    def _cal_returns(self):
        '''
        计算每一组调仓之间的收益
        '''
        if self._group is None: self._rank_and_divide()
              
        self._sub_data['group'] = self._group['group']
        
        sub_data =  self._sub_data
        returns = sub_data.groupby(['date', 'group'])['period_returns'].mean().unstack('group')
        #returns.index = returns.index.get_level_values('date')
        returns = returns.sort_index()
        
        returns.columns = list(range(1, self._group_num + 1))
        returns.columns.name = 'group'
        #returns.columns = [int(i) for i in returns.columns]
        
        returns['long_short'] = - returns[1] + returns[self._group_num]
        if self.IC().iloc[0] < 0:
            returns['long_short'] = -1 * returns['long_short']
        self._returns = returns
        
    def sharpe_ratio(self):
        if self._returns is None: self._cal_returns()
        sr = self._returns.mean() / self._returns.std() * ((252 / self._balance_time) ** 0.5)
        return sr
    
    def turnover(self):
        '''计算换手率'''
        weights = pd.DataFrame(self._group).dropna()
        weights['weights'] = 1
        weights = weights.reset_index().set_index(['code', 'date', 'group'])
        weights /= weights.groupby(['date', 'group']).transform('count')
        weights = weights['weights'].unstack('code').fillna(0)
        weights_shift = weights.groupby('group').shift(1)
        diff = weights - weights_shift
        turn = diff.abs().sum(axis = 1) / 2
        turn = turn.unstack().iloc[1:]
        return (turn[1] + turn[self._group_num]).mean() * 6
    
    def plot_cum(self, **args):
        if self._returns is None: self._cal_returns()
        fig, ax = plt.subplots()
        cum_returns = (self._returns + 1).cumprod()
        cum_returns[cum_returns.columns[:-1]].plot(ax = ax)
        cum_returns['long_short'].plot(ax = ax, linewidth = 3, color = 'blueviolet', alpha = 0.7)
        return ax
    
    def _mdd(self, s):
        s = (s + 1).cumprod()
        max_value = 1
        max_draw_down = 0
        for value in s:
            draw_down = value / max_value - 1
            if draw_down < max_draw_down: max_draw_down = draw_down
            if value > max_value: max_value = value
        return max_draw_down
    
    def _IC(self):
        IC_data = self._sub_data[['period_returns', 'factors']]
        IC_data = IC_data[IC_data['period_returns'] != 0]
        IC_date_group = IC_data.groupby('date')
        Rank_IC_series = IC_date_group.corr(method = 'spearman').iloc[::2, 1]
        IC_series = IC_date_group.corr().iloc[::2, 1]
        IC_data_frame = pd.concat([Rank_IC_series, IC_series], axis = 1)
        IC_data_frame.columns = ['Rank_IC', 'IC']
        IC_data_frame.index = IC_data_frame.index.get_level_values('date')
        IC_data_frame = IC_data_frame.dropna()
        self._IC_data = IC_data_frame
    
    @property
    def daily_returns(self):
        return self._cal_portfolio_returns_between_balancing()
    
    @property
    def returns(self):
        return self._returns
    
    def IC(self):
        if self._IC_data is None:
            self._IC()
        return self._IC_data.mean()
    
    def IR(self, freq = 'M'):
        if self._IC_data is None:
            self._IC()
        IR_data = self._IC_data.mean() / self._IC_data.std()
        IR_data.index = ['Rank_IC_IR', 'IC_IR']
        return IR_data
    
        
    def t_values(self, freq = 'M'):
        t_data = self._sub_data[['returns', 'industry', 'market_value', 'factors']]
        t_data = t_data.dropna()
        def regress_t(data):
            industry_dummies = pd.get_dummies(data['industry'])
            X = pd.concat([industry_dummies, data[['market_value', 'factors']]], axis = 1)
            y = data['returns']
            result = sm.OLS(y, X).fit()
            return result.tvalues[-1]
        t_series = t_data.groupby('date').apply(regress_t)
        #return (t_series > 2).mean()
        t_mean = t_series.mean()
        t_significant_mean = (t_series.abs() > 2).mean()
        significat_subset = t_series[t_series.abs() > 2]
        direction = significat_subset.shift(1) * significat_subset
        t_same_direction = (direction > 0).sum() / len(t_series)
        t_opposite_direction = (direction < 0).sum() / len(t_series)
        return t_mean, t_significant_mean, t_same_direction, t_opposite_direction
    
    def max_drawdown(self):
        return self._returns.fillna(0).apply(self._mdd)
    
    def plot_annual_returns(self, title = None):
         #年化收益率绘图
        returns = self._returns.mean().iloc[:-1] * (252 / self._balance_time)
        fig_returns, ax_returns = plt.subplots()
        title = title if title is not None else 'Annual Returns'
        returns.plot(kind = 'bar', ax = ax_returns, title = title)
        return fig_returns
    
    def plot_sharpe_ratio(self):
        fig_SR, ax_SR = plt.subplots()
        SR = self.sharpe_ratio()
        SR.plot(kind = 'bar', ax = ax_SR, title = 'Sharpe Ratio')
        return fig_SR
        
    def plot_IC(self):
        #绘制IC图
        fig_IC, ax_IC = plt.subplots()
        #index = self._IC_data.index
        #xticks = [index[i].strftime('%Y-%m-%d') if i % 5 == 0 else '' for i in range(len(index))]
        self._IC_data.plot(kind = 'bar', title = 'IC')
        return fig_IC
    
    def plot_IR(self):
        #绘制ICIR图
        IR_by_year = self._IC_data.resample('Y').mean() / self._IC_data.resample('Y').std()
        IR_by_year.index = IR_by_year.index.year
        fig_IR, ax_IR = plt.subplots()
        IR_by_year.plot(kind = 'bar', title = 'IR', figsize = (8, 6))
        return fig_IR
    
    def plot_long_short_value(self, title = None, start_date = None):
        #绘制多空净值图
        returns = self._returns['long_short']
        if start_date is not None:
            returns = returns[start_date:]
        pure_assets = (returns + 1).cumprod()
        fig_value, ax_value = plt.subplots()
        title = title if title is not None else 'Long Short Portfolio Value'
        pure_assets.plot(ax = ax_value, title = title)
        return fig_value
    
    def information_ratio(self, base_returns):
        pass
    
    def summary(self, freq = 'M'):
        
        SR = self.sharpe_ratio()
        print(self.plot_sharpe_ratio())
        
        print(self.plot_annual_returns())
        
        top_SR = SR.iloc[0]
        bottom_SR = SR.iloc[-2]
        if self.IC().iloc[0] > 0:
            top_SR, bottom_SR = bottom_SR, top_SR
        long_short_SR = SR.iloc[-1]
        top_returns_mean = self._returns.mean().iloc[0] * (252 / self._balance_time)
        bottom_returns_mean = self._returns.mean().iloc[-2] * (252 / self._balance_time)
        if self.IC().iloc[0] > 0:
            top_returns_mean, bottom_returns_mean = bottom_returns_mean, top_returns_mean
        market_mean = self._returns.mean().mean() * (252 / self._balance_time)
        long_short_returns_mean = self._returns.mean().iloc[-1] * (252 / self._balance_time)
        Rank_IC, IC = self.IC().values
        Rank_IR, IR = self.IR().values
        max_drawdown = self.max_drawdown().iloc[-1]
        t_mean, greater_than2, t_same_direction, t_opposite_direction = self.t_values()
        turnover = self.turnover()
        win_ratio = (self._returns['long_short'] > 0).mean()
        index_names = ['Top Portfolio Returns Mean', 'Bottom Portfolio Returns Mean',\
                       'Market Returns Mean', 'Long-Short Returns Mean', \
                       'Top Portfolio Sharpe Ratio', 'Long-Short Portfolio Sharpe Ratio', \
                       'Rank_IC', 'IC', 'Rank_IR', 'IR', 'Max Drawdown', 't', \
                       'Greater Than 2', 't_same_direction', 't_opposite_direction',\
                       'Win Ratio', 'Turnover']
        statistics = pd.Series([top_returns_mean, bottom_returns_mean, \
                                market_mean, long_short_returns_mean, \
                                top_SR, long_short_SR, \
                                Rank_IC, IC, Rank_IR, IR, max_drawdown, t_mean,\
                                greater_than2, t_same_direction, t_opposite_direction, \
                                win_ratio, turnover],\
                        index = index_names)
        
        print(self.plot_IC())
        print(self.plot_IR())
        
        print(self.plot_long_short_value())
        
        return statistics