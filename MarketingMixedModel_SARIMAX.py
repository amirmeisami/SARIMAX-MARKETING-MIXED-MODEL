import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, date
import datetime
import psycopg2
import MySQLdb
import pycountry
import pandas.io.sql as psql
from pandas import Series, DataFrame, Panel

class Mixed_model:
    

    def __init__(self, stTrain, endTrain, stPred, endPred, target_country, baseline_country, order,
                 seasonal_order=None, paid=True, baseorg=True, view=True, rank=True):
        
        self.stTrain = stTrain
        self.endTrain = endTrain
        self.stPred = stPred
        self.endPred = endPred
        self.target = target_country
        self.baseline = baseline_country
        self.order = order
        self.seasonal_order = seasonal_order
        self.paid = paid
        self.baseorg = baseorg
        self.view = view
        self.rank = rank
        stT = date(int(self.stTrain.split('-')[0]), int(self.stTrain.split('-')[1]), int(self.stTrain.split('-')[2]))
        endT = date(int(self.endTrain.split('-')[0]), int(self.endTrain.split('-')[1]), int(self.endTrain.split('-')[2]))
        stP = date(int(self.stPred.split('-')[0]), int(self.stPred.split('-')[1]), int(self.stPred.split('-')[2]))
        self.endP = date(int(self.endPred.split('-')[0]), int(self.endPred.split('-')[1]), int(self.endPred.split('-')[2]))
        self.period = (self.endP - stT).days + 1
        self.pred_period = (self.endP - stP).days + 1
        
        
    def source_data(self):
        
        st_date = self.stTrain
#        st_date = '2014-10-1'
        stD = date(int(st_date.split('-')[0]), int(st_date.split('-')[1]), int(st_date.split('-')[2]))
        if self.view and stD < datetime.datetime.strptime('2015-4-1',"%Y-%m-%d").date():
            raise RuntimeError('I know it sucks but we dont have view-count data for anytime before 2015-4-1!')
        if self.view:
            db_red = psycopg2.connect(host="***", database="***", port="***",
                                  user="***", password="***")
            db_red.autocommit = True
            df_red = pd.read_sql('''select date,sum(installs) as install, sum(pageviewcount) as view
                                from appstoredata_itunes_metrics where game='***' 
                                and country='%s' group by date;''' % pycountry.countries.get(alpha2=self.target).name, 
                                con=db_red)  
                            
            df_red['date'] = pd.to_datetime(df_red['date'])
            ts_view_target1 = Series(df_red.view.tolist(), 
                                     index=df_red.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)
            ts_install_target1 = Series(df_red.install.tolist(), 
                                        index=df_red.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)
            if len(ts_view_target1) < (self.endP-stD).days :
                ts_view_target1[pd.to_datetime(st_date)] = 0
                ts_view_target1 = ts_view_target1.resample('D', how='sum')[st_date:self.endPred].fillna(0)
                ts_install_target1[pd.to_datetime(st_date)] = 0
                ts_install_target1 = ts_install_target1.resample('D', how='sum')[st_date:self.endPred].fillna(0)
            ts_view_target = (ts_view_target1)/(ts_view_target1.sum())
            ts_install_target = (ts_install_target1)/(ts_install_target1.sum())
        else:
            ts_view_target = []
            ts_view_target1 = []
            ts_install_target = []  
            ts_install_target1 = []
        
        db = MySQLdb.connect(
        host = '***', 
        user = '***', 
        passwd = '***', 
        db = '***', 
        port = '***')
        
        df_mysql = pd.read_sql('''select metrics_daily.date as date, dim_country.name as country,
                               sum(metrics_daily.value) as value, dim_channel.channel_type as type
                               from metrics_daily left join dim_channel on dim_channel.id = metrics_daily.channel_id 
                               left join dim_country on dim_country.id = metrics_daily.country_id where project_id=195 
                               and metrics_daily.platform_id=2 and metric_id in (5) group by date, type, country;''', con=db)  
                       
        
        df_mysql['date'] = pd.to_datetime(df_mysql['date'])
        all_data_target = df_mysql[df_mysql.country==self.target]
        org_data_target = df_mysql[(df_mysql.type=='ORGANIC') & (df_mysql.country==self.target)]
        ts_org_target1 = Series(org_data_target.value.tolist(), 
                               index=org_data_target.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)
        ts_all_target1 = Series(all_data_target.value.tolist(), 
                                index=all_data_target.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)
        ts_org_target = (ts_org_target1)/(ts_org_target1.sum())
        ts_all_target = (ts_all_target1)/(ts_all_target1.sum())
        
        if self.baseorg:
            org_data_base = df_mysql[(df_mysql.type=='ORGANIC') & (df_mysql.country==self.baseline)]
            ts_org_base1 = Series(org_data_base.value.tolist(), 
                                 index=org_data_base.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)   
            ts_org_base = (ts_org_base1-ts_org_base1.min())/(ts_org_base1.max()-ts_org_base1.min())
        else:
            ts_org_base = []
            ts_org_base1 = []
        
        if self.paid:
            paid_data_target = df_mysql[(df_mysql.type=='PAID') & (df_mysql.country==self.target)]
            ts_paid_target1 = Series(paid_data_target.value.tolist(),
                                    index=paid_data_target.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)
            if len(ts_paid_target1) < (self.endP-stD).days :
                ts_paid_target1[pd.to_datetime(st_date)] = 0
                ts_paid_target1 = ts_paid_target1.resample('D', how='sum')[st_date:self.endPred].fillna(0)
            ts_paid_target = (ts_paid_target1)/(ts_paid_target1.sum())
        else:
            ts_paid_target = []
            ts_paid_target1 = []
            
        if self.rank:
            df_rank = pd.read_sql('''select date, max(1/sqrt(rank)) as bestRank from kabam_ranks_data_free where 
                                    country='%s' and device!='android'and game='***' 
                                    and category='Overall' group by date;''' % self.target, con=db)  
            
            df_rank['date'] = pd.to_datetime(df_rank['date'])
            ts_rank_target1 = Series(df_rank.bestRank.tolist(), 
                                     index=df_rank.date.tolist()).resample('D', how='sum')[st_date:self.endPred].fillna(0)
            if len(ts_rank_target1) < (self.endP-stD).days :
                ts_rank_target1[pd.to_datetime(st_date)] = 0
                ts_rank_target1 = ts_rank_target1.resample('D', how='sum')[st_date:self.endPred].fillna(0)
            ts_rank_target = (ts_rank_target1)/(ts_rank_target1.sum())
        else:
            ts_rank_target = []
            ts_rank_target1 = []
        
#        endog = ts_org_target1
#        endog = ts_install_target
        endog = ts_all_target1
        
        Tlist = [self.paid, self.baseorg, self.view, self.rank]
        dff = DataFrame()
        tList = [ts_paid_target, ts_org_base, ts_view_target, ts_rank_target]
        tlist = ['paid', 'base', 'view', 'rank']
        for i in xrange(0,len(Tlist)):
            if Tlist[i]:
                dff[tlist[i]] = tList[i]
        if dff.empty:
            raise RuntimeError('Where is your exog variable? Do you need a coffee or something?!')
                
        exog = dff
        
        return (endog, exog)
        
    def fit_model(self):
        
        endog, exog = self.source_data()
        if self.seasonal_order != None:
            mod = sm.tsa.statespace.SARIMAX(endog.ix[self.stTrain:self.endTrain], exog = exog.ix[self.stTrain:self.endTrain],
                                            order=self.order, trend = 'c', seasonal_order = self.seasonal_order, enforce_stationarity=False,
                                            enforce_invertibility=False)
        else:
            mod = sm.tsa.statespace.SARIMAX(endog.ix[self.stTrain:self.endTrain], exog = exog.ix[self.stTrain:self.endTrain],
                                            order=self.order, trend = 'c', enforce_stationarity=False, enforce_invertibility=False)
                                            
        fit_res = mod.fit(trend='c', disp=False,transparams=True)
        
        
        if self.seasonal_order != None:
            mod = sm.tsa.statespace.SARIMAX(endog, exog = exog, order=self.order, trend = 'c', 
                                            seasonal_order = self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        else:
            mod = sm.tsa.statespace.SARIMAX(endog, exog = exog, order=self.order, trend = 'c', enforce_stationarity=False, enforce_invertibility=False)
            
        mod.update(fit_res.params)
        res = mod.filter()
        
        return (res, fit_res)
        
    def predict_ts(self):
        
        res, fit_res = self.fit_model()
        predict_res = res.predict(full_results=True)
        predict = predict_res.forecasts
        idx = res.data.predict_dates._mpl_repr()        
        predict_dy_res = res.predict(dynamic=self.period-self.pred_period-1, full_results=True)
        predict_dy = predict_dy_res.forecasts
        cov = predict_res.forecasts_error_cov
        
        # 95% confidence intervals
        critical_value = norm.ppf(1 - 0.05 / 2.)
        std_errors = np.sqrt(cov.diagonal().T)
        ci = np.c_[
            (predict - critical_value*std_errors)[:, :, None],
            (predict + critical_value*std_errors)[:, :, None],
        ]
                
        # Dynamic predictions
        cov_dy = predict_dy_res.forecasts_error_cov
        
        # 95% confidence intervals
        critical_value = norm.ppf(1 - 0.05 / 2.)
        std_errors_dy = np.sqrt(cov_dy.diagonal().T)
        ci_dy = np.c_[
            (predict_dy - critical_value*std_errors_dy)[:, :, None],
            (predict_dy + critical_value*std_errors_dy)[:, :, None],
        ]        
        
        return (ci, predict, idx, ci_dy, predict_dy)
        
    def plot_ts(self):
        
        ci, predict, idx, ci_dy, predict_dy = self.predict_ts()
        endog, exog = self.source_data()

        fig, ax = plt.subplots(figsize=(11,7))
        npre = 7
        plt.title('%s Organic MMM' % self.target, fontsize=18)
        plt.ylabel('Organic Installs', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        dates = pd.date_range(self.stTrain, self.endPred, freq='D')
        ax.plot(dates, endog[-self.period:], color="#3F5D7D",linewidth=2, label='Observed')
        
        
        ax.plot(idx[-self.pred_period-npre:], predict_dy[0, -self.pred_period-npre:], 'y',linewidth=1.5, label='Dynamic forecast')
        ax.plot(idx[-self.pred_period-npre:], ci_dy[0, -self.pred_period-npre:], 'y--', alpha=0.3)
        

        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False) 
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
        
        legend = ax.legend(loc='upper left',fontsize=12)
        legend.get_frame().set_facecolor('w')
        
    def mape(self):
        
        res, fit_res = self.fit_model()
        endog, exog = self.source_data()
        ci, predict, idx, ci_dy, predict_dy = self.predict_ts()
        print "Model Params:"
        print fit_res.params
        print 'MAPE =', np.mean(np.abs((endog.iloc[-self.pred_period:] - predict_dy[0, -self.pred_period:])
                                        / endog.iloc[-self.pred_period:])) * 100
        
        

m = Mixed_model(stTrain='2015-5-15', endTrain='2015-6-28', stPred='2015-6-29', endPred='2015-7-5', target_country='US',
                baseline_country='CA', order=(1,0,1), seasonal_order=None, paid=True, baseorg=True, view=False, rank=False)
                 
m.plot_ts()
m.mape()
#endog, exog= m.source_data()