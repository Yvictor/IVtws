import os
import time
import signal
import numpy as np
import pandas as pd
from datetime import date,datetime,timedelta
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from bs4 import BeautifulSoup as BS
### use in jupyter Notebook
#from ipywidgets import interactive,IntSlider,FloatSlider,Dropdown,Button,fixed,HBox,VBox,Layout,Play,jslink
from IPython.display import display,clear_output,HTML
from plotly.tools import mpl_to_plotly
from plotly.offline import iplot,iplot_mpl
import bqplot as bq
from colour import Color

def Vol_conversion(input_ele):
    if input_ele == '--':
        return 0
    if input_ele!= '--':
        return int(''.join(input_ele.split(',')))
def mon_float(input_ele):
    if input_ele == '--':
        return None
    if input_ele!= '--':
        return float(''.join(input_ele.split(',')))
def mon_conversion(input_ele):
    if input_ele == '--':
        return 0
    if input_ele!= '--':
        return float(''.join(input_ele.split(',')))
def divin(x,y):
    if x-y>0:
        return x-y
    else:
        return 0
def div(c0,c1):
    if c0!=0:
        return c0-c1
    else:
        return 0
def bs_call(S,X,T,rf,sigma):
    from scipy import log,exp,sqrt,stats
    d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    bsc = S*stats.norm.cdf(d1)-X*exp(-rf*T)*stats.norm.cdf(d2)
    return bsc
def bs_put(S,X,T,rf,sigma):
    from scipy import log,exp,sqrt,stats
    d1=(log(S/X)+(rf+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    bsp = X*exp(-rf*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)
    return bsp

def crmt(x):
    todayw = date.isoweekday(date.today())
    if todayw >= 4:
        lefttime_today = 8 - todayw
    else:
        lefttime_today = 3 - todayw
    if todayw == 3 and datetime.now()>datetime.now().replace(hour=13,minute=30,second=0,microsecond=0):
        lefttime_today = 5
    current = datetime.today()
    currentdayst = current.replace(hour=8,minute=45,second=0, microsecond=0)
    currentdayend = current.replace(hour=13, minute=45, second=0, microsecond=0)
    if type(x)!=pd.tslib.NaTType:
        rmt = (currentdayend-x) /(currentdayend-currentdayst)
        lefttime_today=lefttime_today+rmt
        return lefttime_today


def implied_vol_call_min(S,X,T,r,c):
    from scipy import log,exp,sqrt,stats
    implied_vol= 1.0;min_value=100.0;tts = 1;co = 1
    while co<=6:
        tts = tts*0.1
        co = co+ 1
        sc = implied_vol
        for i in range(0,10,1):
            sigma = round(sc-tts*(i),3+co)
            d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
            d2 = d1-sigma*sqrt(T)
            call=S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)
            abs_diff=call-c
            if abs_diff > 0 and abs_diff<=min_value:
                min_value=abs_diff;implied_vol=sigma;k=i;call_out=call
    return round(implied_vol*100,2)
def implied_vol_put_min(S,X,T,r,p):
    from scipy import log,exp,sqrt,stats
    implied_vol=1.0;min_value=100.0;tts = 1;co = 1
    while co<=6:
        tts = tts*0.1
        co = co+ 1
        sc = implied_vol
        for i in range(0,10,1):
            sigma = round(sc-tts*(i),3+co)
            d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
            d2 = d1-sigma*sqrt(T)
            put=X*exp(-r*T)*stats.norm.cdf(-d2)-S*stats.norm.cdf(-d1)
            abs_diff=put-p
            if abs_diff > 0 and abs_diff<=min_value:
                min_value=abs_diff;implied_vol=sigma;k=i;put_out=put
    return round(implied_vol*100,2)


class IVstream:
    def __init__(self,opet,clost):
        self.timecurr = datetime.now()
        self.opentime = self.timecurr.replace(hour=opet[0], minute=opet[1], second=0, microsecond=0)
        self.closetime = self.timecurr.replace(hour=clost[0], minute=clost[1], second=0, microsecond=0)
        self.options = []
        self.driver = webdriver.PhantomJS()
        self.driverf = webdriver.PhantomJS()
        self.Call = []
        self.future_table = []
        self.cache = None
        self.OptIndx()
    def TWSEquote(self):
        urlfut = 'http://info512.taifex.com.tw/Future/FusaQuote_Norl.aspx?_Category=1'
        resfut = requests.get(urlfut)
        resfut.encoding = 'utf-8'
        soup = BS(resfut.text,"lxml")
        table = pd.read_html(str(soup.select('#divDG')[0]),index_col=0,header=0)[0]
        divdata = table.iloc[0:3].transpose().loc[['成交價']].applymap(lambda x : mon_float(x)) #table.iloc[0:3].transpose().loc[['成交價']].astype('float')
        if divdata['臺指現貨'].values[0]== None:
            divdata['臺指現貨'] = table.iloc[0].loc['參考價']
        return divdata['臺指現貨'].values[0]
    def close_PhantomJS(self):
        self.driver.service.process.send_signal(signal.SIGTERM)
        self.driver.quit()
        self.driverf.service.process.send_signal(signal.SIGTERM)
        self.driverf.quit()
    def OptIndx(self):
        urlexd = 'http://www.taifex.com.tw/chinese/5/OptIndxFSP.asp'
        resex = requests.get(urlexd)
        resex.encoding = 'utf-8'
        exdsoup = BS(resex.text,"lxml")
        self.weekopexdate = pd.read_html(str(exdsoup.select('.table_c')[0]),header=0)[0][['最後結算日','契約  月份', '臺指選擇權  （TXO）']].set_index('最後結算日')
        self.weekopexdate.columns = ['契約月份','最後結算價']
        self.weekopexdate.index = self.weekopexdate.index.to_datetime()
        self.lastexprice = self.weekopexdate.iloc[0].loc['最後結算價']
        return self.lastexprice

    def futureQuote(self):
        self.driverf.get('http://info512.taifex.com.tw/Future/FusaQuote_Norl.aspx')
        time.sleep(0.1)
        soup = BS(self.driverf.page_source,'lxml')
        self.future_table = pd.read_html(str(soup.select('#divDG')[0]),header=0)[0]

    def get_future(self):
        if len(self.future_table)==0:
            self.futureQuote()
        else:
            soup = BS(self.driverf.page_source,'lxml')
            self.future_table = pd.read_html(str(soup.select('#divDG')[0]),header=0)[0]
        return float(self.future_table['成交價'].iloc[1])

    def OptQoutedriver(self,exda):
        #self.driver.implicitly_wait(3)
        self.driver.get('http://info512.taifex.com.tw/Future/OptQuote_Norl.aspx')
        selectbox = webdriver.support.ui.Select(self.driver.find_element_by_name('ctl00$ContentPlaceHolder1$ddlFusa_SelMon'))
        selectbox.all_selected_options
        self.options = [i.text for i in selectbox.options]
        selectbox.select_by_value(self.options[exda])

    def OptQuote(self,exdat,seln,sleept=0.25):
        if seln==1 or len(self.options)==0:
            self.OptQoutedriver(exdat)
        #self.driver.execute_script('ctl00_ContentPlaceHolder1_lbtnRefresh')
        time.sleep(sleept)#self.driver.implicitly_wait(5)
        self.driver.save_screenshot('screen.png')
        opt_ps = self.driver.page_source
        soup = BS(opt_ps,"lxml")
        table = pd.read_html(str(soup.select('#divDG')[0]))[0]
        Call = table[[0,1,2,3,4,5,6]]
        Put = table[[6,7,8,9,10,11,12]]
        Call.columns = Call.loc[0]
        Call = Call[1:]
        Put.columns = Put.loc[0]
        Put = Put[1:]
        Call['總量'] = Call['總量'].map(Vol_conversion)
        Put['總量'] = Put['總量'].map(Vol_conversion)
        Call.insert(3,'成交價',Call['成交'].map(mon_float))
        Put.insert(3,'成交價',Put['成交'].map(mon_float))
        Call['履約價']=Call['履約價'].map(lambda x:int(x))
        Put['履約價']=Put['履約價'].map(lambda x:int(x))
        opc = Call[Call['履約價']==Call['履約價']]
        opp = Put[Put['履約價']==Put['履約價']]
        comb = np.array(opc['履約價'].tolist())+np.array(opc['成交價'].tolist())-np.array(opp['成交價'].tolist())
        instc = pd.Series(comb,index=[opc.index])
        instp = pd.Series(comb,index=[opp.index])
        opc.insert(5,'組合價',instc)
        opp.insert(5,'組合價',instp)
        pdtcin = opc.apply(lambda row: divin(row['組合價'],row['履約價']),axis=1)
        opc.insert(5,'內含價值',pdtcin)
        pdtctv = opc.apply(lambda row: div(row['成交價'],row['內含價值']),axis=1)
        opc.insert(6,'時間價值',pdtctv)
        pdtpin = opp.apply(lambda row: divin(row['履約價'],row['組合價']),axis=1)
        opp.insert(5,'內含價值',pdtpin)
        pdtptv = opp.apply(lambda row: div(row['成交價'],row['內含價值']),axis=1)
        opp.insert(6,'時間價值',pdtptv)
        opc['時間'] = pd.to_datetime(opc['時間'])
        opp['時間'] = pd.to_datetime(opp['時間'])
        opc.insert(10,'TCUL',opc['時間'].map(crmt))
        opp.insert(10,'TCUL',opp['時間'].map(crmt))
        self.Call = opc
        self.Put = opp

    def getStreamQuote(self,seln,exdat):
        if seln==1 or len(self.options)==0:
            self.OptQoutedriver(exdat)
        opt_ps = self.driver.page_source
        soup = BS(opt_ps,"lxml")
        table = pd.read_html(str(soup.select('#divDG')[0]))[0]
        Call = table[[0,1,2,3,4,5,6]]
        Put = table[[6,7,8,9,10,11,12]]
        Call.columns = Call.loc[0]
        Call = Call[1:]
        Put.columns = Put.loc[0]
        Put = Put[1:]
        Call['總量'] = Call['總量'].map(Vol_conversion)
        Put['總量'] = Put['總量'].map(Vol_conversion)
        Call.insert(3,'成交價',Call['成交'].map(mon_float))
        Put.insert(3,'成交價',Put['成交'].map(mon_float))
        Call['履約價']=Call['履約價'].map(lambda x:int(x))
        Put['履約價']=Put['履約價'].map(lambda x:int(x))
        opc = Call[Call['履約價']==Call['履約價']]
        opp = Put[Put['履約價']==Put['履約價']]
        comb = np.array(opc['履約價'].tolist())+np.array(opc['成交價'].tolist())-np.array(opp['成交價'].tolist())
        instc = pd.Series(comb,index=[opc.index])
        instp = pd.Series(comb,index=[opp.index])
        opc.insert(5,'組合價',instc)
        opp.insert(5,'組合價',instp)
        pdtcin = opc.apply(lambda row: divin(row['組合價'],row['履約價']),axis=1)
        opc.insert(5,'內含價值',pdtcin)
        pdtctv = opc.apply(lambda row: div(row['成交價'],row['內含價值']),axis=1)
        opc.insert(6,'時間價值',pdtctv)
        pdtpin = opp.apply(lambda row: divin(row['履約價'],row['組合價']),axis=1)
        opp.insert(5,'內含價值',pdtpin)
        pdtptv = opp.apply(lambda row: div(row['成交價'],row['內含價值']),axis=1)
        opp.insert(6,'時間價值',pdtptv)
        opc['時間'] = pd.to_datetime(opc['時間'])
        opp['時間'] = pd.to_datetime(opp['時間'])
        opc.insert(10,'TCUL',opc['時間'].map(crmt))
        opp.insert(10,'TCUL',opp['時間'].map(crmt))
        self.Call = opc
        self.Put = opp
    def getOptable(self,exdat=1,seln=1,typ='Call',risk_free_rate = 0.0136):
        if seln ==1 or seln==2 or len(self.Call)==0:
            self.get_future()
            self.getStreamQuote(seln,exdat)
            self.Call = self.Call.dropna()
            opciv = self.Call.apply(lambda row: implied_vol_call_min(row['組合價'],row['履約價'],self.Call.TCUL.iloc[0]/255,risk_free_rate,row['成交價']),axis=1)
            self.Call.insert(7,'隱含波動率',opciv)
            self.Put = self.Put.dropna()
            oppiv = self.Put.apply(lambda row:implied_vol_put_min(row['組合價'],row['履約價'],self.Put.TCUL.iloc[0]/255,risk_free_rate,row['成交價']),axis=1)
            self.Put.insert(7,'隱含波動率',oppiv)
            self.Callless = self.Call[(self.Call['隱含波動率']!=100)&(self.Call['隱含波動率']!=0)&(self.Call['成交價']>0.5)&(self.Call['總量']>100)]
            self.Putless = self.Put[(self.Put['隱含波動率']!=100)&(self.Put['隱含波動率']!=0)&(self.Put['成交價']>0.5)&(self.Put['總量']>100)]

            #self.Callless.loc[:,:"總量"] = self.Callless.loc[:,:"總量"].applymap(float)
            #self.Putless.loc[:,:"總量"] = self.Putless.loc[:,:"總量"].applymap(float)
        #elif seln!=1 and len(self.Call)==0:

        if typ=='Call':
            return self.Call
        elif typ=='Put':
            return self.Put

    def update_data(self,exdat=1,seln=2,risk_free_rate = 0.0136):
        if seln ==1 or seln==2 or len(self.Call)==0:
            self.get_future()
            self.getStreamQuote(seln,exdat)
            self.Call = self.Call.dropna()
            opciv = self.Call.apply(lambda row: implied_vol_call_min(row['組合價'],row['履約價'],self.Call.TCUL.iloc[0]/255,risk_free_rate,row['成交價']),axis=1)
            self.Call.insert(7,'隱含波動率',opciv)
            self.Put = self.Put.dropna()
            oppiv = self.Put.apply(lambda row:implied_vol_put_min(row['組合價'],row['履約價'],self.Put.TCUL.iloc[0]/255,risk_free_rate,row['成交價']),axis=1)
            self.Put.insert(7,'隱含波動率',oppiv)
            self.Callless = self.Call[(self.Call['隱含波動率']!=100)&(self.Call['隱含波動率']!=0)&(self.Call['成交價']>0.5)&(self.Call['總量']>100)]
            self.Putless = self.Put[(self.Put['隱含波動率']!=100)&(self.Put['隱含波動率']!=0)&(self.Put['成交價']>0.5)&(self.Put['總量']>100)]
    def bqstreamplot(self):
        Cxs = bq.DateScale()
        Cys = bq.LinearScale()
        Cx = self.CallIVtable.index.values
        Cy = self.CallIVtable.as_matrix().transpose()
        Ccol = self.CallIVtable.columns.tolist()

        self.Cline = bq.Lines(x=Cx, y=Cy, scales={'x': Cxs, 'y': Cys},
                        colors=[i.hex for i in list(Color(rgb=(0.95,0,0)).range_to(Color(rgb=(0.45,0.1,0)), len(Ccol)))],
                        labels=Ccol,
                        enable_hover=True,
                        display_legend=True)
        Cxax = bq.Axis(scale=Cxs, label='Datetime', grid_lines='solid')
        Cyax = bq.Axis(scale=Cys, orientation='vertical', tick_format='0.1f', label='CallIV', grid_lines='solid')
        figC = bq.Figure(marks=[self.Cline], axes=[Cxax, Cyax], animation_duration=1000)

        Pxs = bq.DateScale()
        Pys = bq.LinearScale()
        Px = self.PutIVtable.index.values
        Py = self.PutIVtable.as_matrix().transpose()
        Pcol = self.PutIVtable.columns.tolist()

        self.Pline = bq.Lines(x=Px, y=Py, scales={'x': Pxs, 'y': Pys},
                        colors=[i.hex for i in list(Color(rgb=(0,0.75,0)).range_to(Color(rgb=(0,0,0.45)), len(Pcol)))],
                        labels=Pcol,
                        enable_hover=True,
                        display_legend=True)
        Pxax = bq.Axis(scale=Pxs, label='Datetime', grid_lines='solid')
        Pyax = bq.Axis(scale=Pys, orientation='vertical', tick_format='0.1f', label='PutIV', grid_lines='solid')
        figP = bq.Figure(marks=[self.Pline], axes=[Pxax, Pyax], animation_duration=1000)
        display(HBox(([figC,figP])))


    def init_table(self,select_settled=0):
        self.CallIVtable = pd.DataFrame()
        self.PutIVtable = pd.DataFrame()
        self.update_data(exdat=select_settled,seln=2)
        CallIV = self.Callless[['履約價','隱含波動率','時間']].copy()
        CallIV.loc[:,'履約價']=self.Callless['履約價'].map(lambda x:'Call_'+str(x))
        PutIV = self.Putless[['履約價','隱含波動率','時間']].copy()
        PutIV.loc[:,'履約價']=self.Putless['履約價'].map(lambda x:'Put_'+str(x))

        CallIV = CallIV[['履約價','隱含波動率']].set_index('履約價').transpose()
        CallIV.index =pd.Index([self.Callless['時間'].max().to_datetime()], name='datetime')
        PutIV = PutIV[['履約價','隱含波動率']].set_index('履約價').transpose()
        PutIV.index =pd.Index([self.Putless['時間'].max().to_datetime()], name='datetime')

        self.CallIVtable = self.CallIVtable.append(CallIV)
        self.PutIVtable = self.PutIVtable.append(PutIV)
        #self.bqstreamplot()

    def append_IV(self):
        record_time = datetime.now()
        createfig = 0
        while datetime.now()> self.opentime and datetime.now() < self.closetime:
            self.update_data(exdat=0,seln=2)
            clear_output()
            display(self.Callless,self.Putless)
            time.sleep(5)
            CallIV = self.Callless[['履約價','隱含波動率','時間']].copy()
            CallIV.loc[:,'履約價']=self.Callless['履約價'].map(lambda x:'Call_'+str(x))
            PutIV = self.Putless[['履約價','隱含波動率','時間']].copy()
            PutIV.loc[:,'履約價']=self.Putless['履約價'].map(lambda x:'Put_'+str(x))

            CallIV = CallIV[['履約價','隱含波動率']].set_index('履約價').transpose()
            CallIV.index =pd.Index([self.Callless['時間'].max().to_datetime()], name='datetime')
            PutIV = PutIV[['履約價','隱含波動率']].set_index('履約價').transpose()
            PutIV.index =pd.Index([self.Putless['時間'].max().to_datetime()], name='datetime')

            self.CallIVtable = self.CallIVtable.append(CallIV)
            self.PutIVtable = self.PutIVtable.append(PutIV)
            #update plot
            #self.Cline.x = self.CallIVtable.index.values
            #self.Cline.y = self.CallIVtable.as_matrix().transpose()
            #self.Cline.colors = [i.hex for i in list(Color(rgb=(0.95,0,0)).range_to(Color(rgb=(0.45,0.1,0)), len(self.CallIVtable.columns.tolist())))]

            #self.Pline.x = self.PutIVtable.index.values
            #self.Pline.y = self.PutIVtable.as_matrix().transpose()
            #self.Pline.colors = [i.hex for i in list(Color(rgb=(0,0.75,0)).range_to(Color(rgb=(0,0,0.45)), len(self.PutIVtable.columns.tolist())))]

            #save table
            if datetime.now() - record_time>=timedelta(minutes=5):
                record_time = datetime.now()
                self.CallIVtable.to_csv('CallIV.csv',encoding='utf-8')
                self.PutIVtable.to_csv('PutIV.csv',encoding='utf-8')

    def creatSTwithPlot(self,futshare,Cal1,c1share,Cal2,c2share,Put1,p1share,Put2,p2share,showrange,up,down,customcur,risk_free_rate=0.0136):
        stcom = []
        Cal1=int(Cal1)
        Cal2=int(Cal2)
        Put1=int(Put1)
        Put2=int(Put2)
        if futshare!=0:
            st = ['future',self.get_future(),0,futshare]
            stcom.append(st)
        if c1share!=0:
            st = ['Call',Cal1,
                  float(self.Callless[self.Callless['履約價']==Cal1]['成交'].values[0]),
                  self.Callless[self.Callless['履約價']==Cal1]['組合價'].values[0],
                  c1share]
            stcom.append(st)
        if c2share!=0:
            st = ['Call',Cal2,
                  float(self.Callless[self.Callless['履約價']==Cal2]['成交'].values[0]),
                  self.Callless[self.Callless['履約價']==Cal2]['組合價'].values[0],
                  c2share]
            stcom.append(st)
        if p1share!=0:
            st = ['Put', Put1,
                  float(self.Putless[self.Putless['履約價']==Put1]['成交'].values[0]),
                  self.Putless[self.Putless['履約價']==Put1]['組合價'].values[0],
                  p1share]
            stcom.append(st)
        if p2share!=0:
            st = ['Put',Put2,
                  float(self.Putless[self.Putless['履約價']==Put2]['成交'].values[0]),
                  self.Putless[self.Putless['履約價']==Put2]['組合價'].values[0],
                  p2share]
            stcom.append(st)
        print(stcom)
        if len(stcom)>0:
            lastEXP = self.lastexprice
            twsecurrent = self.TWSEquote()
            TWSErange = np.arange(lastEXP-showrange,lastEXP+showrange,10)
            Tcul = self.Call.dropna().TCUL.iloc[0]/255#self.Callless['TCUL'].iloc[3]/255
            #risk_free_rate = self.risk_free_rate
            futureprofit = []
            Callprofit = []
            Putprofit = []
            Callintimeprofit = []
            Putintimeprofit = []
            Calltimectrl =[]
            Puttimectrl = []
            for s in stcom:
                if s[0] == 'future':
                    futureprofit.append((TWSErange-s[1])*s[3])
                if s[0] =='Call':
                    Callprofit.append(((abs(TWSErange-s[1])+TWSErange-s[1])/2-s[2])*s[4])
                    IV = self.Call[self.Call['履約價']==s[1]]['隱含波動率'].values[0]/100
                    Callintimeprofit.append((bs_call(TWSErange,s[1],Tcul,risk_free_rate,IV)-s[2])*s[4])
                    Calltimectrl.append((bs_call(TWSErange,s[1],Tcul*customcur,risk_free_rate,IV)-s[2])*s[4])

                if s[0] =='Put':
                    Putprofit.append(((abs(s[1]-TWSErange)-TWSErange+s[1])/2-s[2])*s[4])
                    IV = self.Put[self.Put['履約價']==s[1]]['隱含波動率'].values[0]/100
                    Putintimeprofit.append((bs_put(TWSErange,s[1],Tcul,risk_free_rate,IV)-s[2])*s[4])
                    Puttimectrl.append((bs_put(TWSErange,s[1],Tcul*customcur,risk_free_rate,IV)-s[2])*s[4])
                STprofit = 0
                STtime = 0
                customcurve = 0

            if len(futureprofit)!=0:
                for f in futureprofit:
                    STprofit = STprofit + f
                    STtime = STtime + f
                    customcurve = customcurve + f
            if len(Callprofit)!=0:
                for Call in Callprofit:
                    STprofit = STprofit + Call
            if len(Putprofit)!=0:
                for Put in Putprofit:
                    STprofit = STprofit + Put

            if len(Callintimeprofit)!=0:
                for Callt in Callintimeprofit:
                    STtime = STtime + Callt
            if len(Putintimeprofit)!=0:
                for Putt in Putintimeprofit:
                    STtime = STtime + Putt

            if len(Calltimectrl)!=0:
                for Callti in Calltimectrl:
                    customcurve = customcurve + Callti
            if len(Puttimectrl)!=0:
                for Putti in Puttimectrl:
                    customcurve = customcurve + Putti
            lastexpX = [lastEXP,lastEXP]
            lastexpY = [-down,up]
            twseX = [twsecurrent,twsecurrent]
            futureX = [self.get_future(),self.get_future()]
            twseY = [-down+40,up-40]
            opX = np.array([s[3] for s in stcom if s[0]!='future']).mean()
            optionX = [opX, opX]
            Xra = 300
            lastexpXup = [lastEXP+Xra,lastEXP+Xra]
            lastexpXdown = [lastEXP-Xra,lastEXP-Xra]
            fig = plt.figure(figsize=(16*3/4,9*3/4))
            ax = fig.add_subplot(1,1,1)
            ax.plot(TWSErange,np.zeros(len(TWSErange)),'gray',label='zeros')
            ax.plot(lastexpX,lastexpY,color=(0.2,0.2,0.7),label='last_exp')
            ax.plot(lastexpXup,lastexpY,color=(0.2,0.7,0.2),label='exp_up')
            ax.plot(lastexpXdown,lastexpY,color=(0.2,0.7,0.2),label='exp_down')
            ax.plot(twseX,twseY,'y',label='twse_current')
            ax.plot(futureX,twseY,'c',label='future_current')
            ax.plot(optionX,twseY,'r',label='option_current')
            ax.plot(TWSErange,STprofit,label='EXdcurve',color = 'k',linewidth=1.5 )
            ax.plot(TWSErange,STtime,label='todaycurve',color = 'dimgray')
            ax.plot(TWSErange,customcurve,label='customcurve',color = (0.3,0.6,0.6))
            #ax.legend()
            ax.set_xlim(lastEXP-showrange,lastEXP+showrange)
            ax.grid()
            ax.set_xlabel('Stock price')
            ax.set_ylabel('Profit (loss)')
            #plt.show()
            iplot_mpl(fig)
