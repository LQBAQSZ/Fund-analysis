#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols#导数据


# In[2]:


import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=False)


# In[3]:


from WindPy import *
 
w.start()

from WindPy import *
from sqlalchemy import create_engine
import datetime,time

import datetime as dt
import time
from datetime import *   


# # 1. 导数据

# In[4]:


df1 = w.wsd("001825.OF", "nav", "2018-03-01", "2021-08-31", "",usedf = True)
df1 = df1[1].reset_index()
df1.columns=['date','Fund']
#不管数据类型如何，先转换一下。
df1['date']= pd.to_datetime(df1['date'])

#转换各列的格式，从str转成float64，要不然后边不好算.

df1['Fund'] = df1['Fund'].astype('float64')


# In[5]:


#导无风险利率数据：SHIBOR隔夜拆借。

df2 = w.edb("M0017138", "2018-03-01", "2021-08-31","Fill=Previous",usedf = True)
df2 = df2[1].reset_index()


# In[6]:


df2['index']= pd.to_datetime(df2['index'])


# In[7]:


df2.columns=['date','SHIBOR']
df2['SHIBOR'] = df2['SHIBOR'].astype('float64')
df2["SHIBOR"]=df2["SHIBOR"]/100


# In[8]:


datasetm1 = w.wsd("H00906.CSI", "close,pct_chg", "ED-4Y", "2021-08-31", "",usedf = True)
datasetm1 = datasetm1[1].reset_index()


# In[9]:


datasetm2 = w.wsd("H11001.CSI", "close,pct_chg", "ED-4Y", "2021-08-31", "",usedf = True)
datasetm2 = datasetm2[1].reset_index()


# In[10]:


datasetm1['index']= pd.to_datetime(datasetm1['index'])
datasetm2['index']= pd.to_datetime(datasetm2['index'])


# In[11]:


datasetm = pd.merge(datasetm1,datasetm2,how='left',on = 'index')
datasetm.head(5)


# In[12]:


datasetm.columns = ["date","H00906.CSI","ret_H00906.CSI","H11001.CSI","ret_H11001.CSI"]
datasetm["ret_H00906.CSI"] = datasetm["ret_H00906.CSI"]/100
datasetm["ret_H11001.CSI"] = datasetm["ret_H11001.CSI"]/100

datasetm["ret_m"] = datasetm["ret_H00906.CSI"]*0.8+datasetm["ret_H11001.CSI"]*0.2

datasetm['date']= pd.to_datetime(datasetm['date'])


# In[13]:


dataset1 = pd.merge(df1,df2,how='left',on = 'date')


# In[14]:


#合并数据
dataset = pd.merge(dataset1,datasetm,how='left',on = 'date')


# In[15]:


#算涨跌幅。

dataset = dataset.reset_index(drop = True)
dataset['Fund_ret'] = (dataset['Fund']-dataset['Fund'].shift(1))/(dataset['Fund'].shift(1))
dataset["accret_m"] = dataset["H00906.CSI"]/dataset["H00906.CSI"][0]*0.8+dataset["H11001.CSI"]/dataset["H11001.CSI"][0]*0.2
dataset.head(10)


# In[16]:


#画基金的走势图，还可以对比各种指数，直接往里面加就好了。
#可以直接在图上拖动，进行缩放。双击返回。

data = [
    
    go.Scatter(
        x=dataset['date'],
        y=dataset['Fund'],
        name = 'Fund'),
 
    go.Scatter(
        x=dataset['date'],
        y=dataset['accret_m'],
        name = 'Fund业绩基准'),
       
        
        ]
    
        
layout = go.Layout(
    title = '基金净值走势'
)
 
fig = go.Figure(data = data,layout=layout)

#让横坐标只显示10个


iplot(fig)


# # 4.持仓分析

# ## 4.1 持仓

# In[17]:


a = w.wset("allfundhelddetail","rptdate=20190630;windcode=001825.OF;field=sec_name,rpt_date,stock_code,stock_name,proportiontototalstockinvestments,proportiontonetvalue",usedf = True)[1]


# In[18]:


a.columns = ["基金名称","报告期","股票代码","股票简称","占股票投资比例","占基金净值比重"]
a["占股票投资比例"] = a["占股票投资比例"]/100
a["占基金净值比重"] = a["占基金净值比重"]/100


# In[19]:


code=list(a["股票代码"])


# In[20]:


a


# ## 4.1.1 持仓明细分析

# In[21]:


c = w.wss(code, "sec_name,pe,pb,pcf_ncf,dividendyield2,roe_basic,fa_orgr_ttm,fa_gpmgr_ttm",
      "tradeDate=20190930;ruleType=8;rptDate=20190930",usedf = True)[1]
c.columns = ["股票简称","市盈率","市净率","市现率","股息率（近12个月）","ROE","营业收入增长率TTM","毛利率增长率TTM"]


# In[22]:


df41 = pd.merge(a,c,on = '股票简称')


# In[23]:


df41


# ## 4.1.2 行业分析

# In[24]:


b = w.wss(code, "pct_chg_nd,industry_citic,,indexcode_citic",
          "days=-30;tradeDate=20190930;industryType=1",usedf = True)[1]


# In[25]:


b = b.reset_index()


# In[26]:


b.columns = ["股票代码","区间收益（季度）","行业（中信一级分类）","行业代码"]


# In[27]:


code_ind=list(set(b["行业代码"]))#加一层set对原来的list进行去重，wind不支持对重复的数据取数


# In[28]:


b_ind = w.wss(code_ind, "pct_chg_nd",
          "days=-30;tradeDate=20190930",usedf = True)[1]/100


# In[29]:


b_ind = b_ind.reset_index()


# In[30]:


b_ind.columns = ["行业代码","行业区间收益"]


# In[31]:


b.iloc[:,1] = b.iloc[:,1]/100


# In[32]:


dataset4 = pd.merge(a,b,on = '股票代码')


# In[33]:


dataset4


# In[34]:


dataset4t= pd.merge(dataset4,b_ind,how = "left",on = "行业代码")


# In[35]:


dataset4t["股票超行业收益"] = dataset4t["区间收益（季度）"]-dataset4t["行业区间收益"]
dataset4t = dataset4t.sort_values(by = ["股票超行业收益"],ascending = False)


# In[80]:


dataset4t["fund_weighted_ret"] = dataset4t["占基金净值比重"]*dataset4t["区间收益（季度）"]
dataset4t["stock_weighted_ret"] = dataset4t["占股票投资比例"]*dataset4t["区间收益（季度）"]
dataset4t["stock_weighted_ind_ret"] = dataset4t["占股票投资比例"]*dataset4t["行业区间收益"]
dataset4t = dataset.reset_index(drop = True)
dataset4t


# In[37]:


df42 = dataset4t.groupby("行业（中信一级分类）").agg({'占股票投资比例':'sum','stock_weighted_ret':'sum','stock_weighted_ind_ret':'sum'})
df42 = df42.sort_values(by = ["占股票投资比例"],ascending = False)
df42 = df42.reset_index()
df42["select_stock_ret"] = df42["stock_weighted_ret"]-df42["stock_weighted_ind_ret"]


# In[38]:


df42 = df42.sort_values(by = ["占股票投资比例"],ascending = False)


# In[39]:


pyplt = py.offline.plot
dataz = [go.Bar(
            x=df42["行业（中信一级分类）"],
            y=df42["占股票投资比例"],
            width = 0.4,
)]
layout = go.Layout(
            title = '持股行业占比',
  
)
figure = go.Figure(data = dataz, layout = layout)
iplot(figure)


# In[40]:


df42["select_stock_ret"] = df42["stock_weighted_ret"]-df42["stock_weighted_ind_ret"]
df42


# In[41]:


df42 = df42.sort_values(by = ["select_stock_ret"],ascending = True)


# In[42]:


trace0 = go.Bar(
    y=df42["行业（中信一级分类）"],
    x=df42["stock_weighted_ret"],
    orientation="h",
    name = "持仓加权后基金持股收益"
)

trace1 = go.Bar(
            y=df42["行业（中信一级分类）"],
            x=df42["stock_weighted_ind_ret"], 
            orientation="h",
            name = '持仓加权后中信一级行业指数收益'
)


axis_template=dict(
    showgrid=True,  #网格
    zeroline=False,  #是否显示基线,即沿着(0,0)画出x轴和y轴
    showline=False,

   
)

layout1=go.Layout(
    xaxis=axis_template,
    yaxis=axis_template,
    title = "择股能力分析（按照获取超额收益排名）",
   
)

figure = go.Figure(data=[trace1, trace0],layout = layout1)


figure.show()



# ## 4.1.3 整体配置能力分析

# In[43]:


fund_return = dataset4t["fund_weighted_ret"].sum()
stock_return = dataset4t["stock_weighted_ret"].sum()


# In[44]:


c = w.wsd("001825.OF", "return_q", "2019-06-30", "2019-09-30", "Period=Q",usedf = True)[1].iat[0,0]/100 #基金的区间收益
d = w.wss("H11001.CSI", "pct_chg","tradeDate=2019-09-30;cycle=Q",usedf = True)[1].iat[0,0]/100 #中证全债
# 此处是否年化已经确认，数据没错。
#得加一下市场的中位数或者是排名。


# In[45]:


print(c,d)


# In[46]:


f = (c-0.2*d)/fund_return-1
f


# # 1.收益率、波动率、夏普比率、最大回撤

# ## 1.1年化收益率、波动率、超额收益率、夏普比率

# In[47]:


#合计（后续补到上面表格里）
datasetA = dataset.copy()
year_retA = datasetA['Fund_ret'].mean()*250


year_exretA = (datasetA['Fund_ret']-datasetA['ret_m']).mean()*250  

year_stdretA = datasetA['Fund_ret'].std()*math.sqrt(250)

year_sharperatioA = year_exretA/year_stdretA

year_retA = "%.4f%%" % (year_retA * 100)
year_exretA = "%.4f%%" % (year_exretA * 100)
year_stdretA = "%.4f%%" % (year_stdretA * 100)
year_sharperatioA = "%.4f" % (year_sharperatioA)
total_retA = "%.4f%%" % (dataset["Fund"].values[-1]*100)
total_exret = "%.4f%%" % (dataset["Fund"].values[-1]*100-dataset["accret_m"].values[-1]*100)


print("自成立以来累计总收益：",total_retA)
print("自成立以来累积超额总收益：",total_exret)

print("年化收益率：",year_retA)
print("年化超额收益率",year_exretA)
print("年化标准差：",year_stdretA)
print("夏普率：",year_sharperatioA)


# In[48]:


datasetA['year'] = datasetA['date'].apply(lambda x:pd.to_datetime(x).year)

year_retA = datasetA.groupby("year").agg({'Fund_ret':'mean'})*250 
year_stdretA = datasetA.groupby("year").agg({'Fund_ret':'std'})*math.sqrt(250)

year_datasetA = year_retA.merge(year_stdretA,how = 'left',on='year')
year_datasetA.columns=['year_return','year_std']

year_datasetA["year_returnm"] = datasetA.groupby("year").agg({'ret_m':'mean'})*250

year_datasetA["sharpe_ratio"] = (year_datasetA["year_return"]-year_datasetA["year_returnm"])/year_datasetA["year_std"]

year_datasetA["year_return"] = year_datasetA["year_return"].apply(lambda x: format(x, '.2%'))
year_datasetA["year_std"] = year_datasetA["year_std"].apply(lambda x: format(x, '.2%'))
year_datasetA["year_returnm"] = year_datasetA["year_returnm"].apply(lambda x: format(x, '.2%'))
year_datasetA["sharpe_ratio"] = year_datasetA["sharpe_ratio"].apply(lambda x: format(x, '.2'))



print("notes：最新一年的年度数据并未进行年化")
year_datasetA


# ## 1.3 季胜率、半年胜率、一年胜率

# In[ ]:





# In[49]:


f = lambda x :'%.2f%%'  %  (x*100)

df42.ix[:,1:] = df42.ix[:,1:].applymap(f)
df42


# In[50]:


#再加一个不要超额的收益率。

#半年胜率，年度胜率。
quarter_retA = (datasetA["Fund"]-datasetA["Fund"].shift(60))/datasetA["Fund"].shift(60)
quarter_retA = quarter_retA.dropna() #删除空值
quarter_accretA = (datasetA["accret_m"]-datasetA["accret_m"].shift(60))/datasetA["accret_m"].shift(60)
quarter_accretA = quarter_accretA.dropna()
quarter_exretA = quarter_retA - quarter_accretA

half_retA = (datasetA["Fund"]-datasetA["Fund"].shift(125))/datasetA["Fund"].shift(125)
half_retA = half_retA.dropna() #删除空值
half_accretA = (datasetA["accret_m"]-datasetA["accret_m"].shift(60))/datasetA["accret_m"].shift(60)
half_accretA = half_accretA.dropna()
half_exretA = half_retA - half_accretA

year_retA = (datasetA["Fund"]-datasetA["Fund"].shift(250))/datasetA["Fund"].shift(250)
year_retA = year_retA.dropna() #删除空值
year_accretA = (datasetA["accret_m"]-datasetA["accret_m"].shift(60))/datasetA["accret_m"].shift(60)
year_accretA = year_accretA.dropna()
year_exretA = year_retA-year_accretA


# In[51]:


res11 = list(filter(lambda x: x > 0, quarter_retA))
quarterwin_ratio1 = len(res11)/len(quarter_exretA)

res1 = list(filter(lambda x: x > 0, quarter_exretA))
quarterwin_ratio = len(res1)/len(quarter_exretA)

res21 = list(filter(lambda x: x > 0, half_retA))
halfwin_ratio1 = len(res21)/len(half_exretA)

res2 = list(filter(lambda x: x > 0, half_exretA))
halfwin_ratio = len(res2)/len(half_exretA)

res31 = list(filter(lambda x: x > 0, year_retA))
yearwin_ratio1 = len(res31)/len(year_exretA)

res3 = list(filter(lambda x: x > 0, year_exretA))
yearwin_ratio = len(res3)/len(year_exretA)



name_dict = {'统计频率':[
                    '季度',
                    '半年度',  
                    '年度']}
df_winratio = pd.DataFrame(name_dict)

df_winratio["超额胜率"] = [quarterwin_ratio,halfwin_ratio,yearwin_ratio]
df_winratio["胜率"] = [quarterwin_ratio1,halfwin_ratio1,yearwin_ratio1]


df_winratio["胜率"] = df_winratio["胜率"].apply(lambda x: format(x, '.2%'))
df_winratio["超额胜率"] = df_winratio["超额胜率"].apply(lambda x: format(x, '.2%'))

df_winratio



# In[52]:


PGHH = w.edb("885004.WI", "2018-03-01", "2021-08-31","Fill=Previous",usedf = True)
PGHH = PGHH[1].reset_index()
PGHH['index']= pd.to_datetime(PGHH['index'])
PGHH["CLOSE"] = PGHH["CLOSE"].astype('float64')


# In[53]:


#再加一个不要超额的收益率。

#半年胜率，年度胜率。
quarter_retA = (PGHH["CLOSE"]-PGHH["CLOSE"].shift(60))/PGHH["CLOSE"].shift(60)
quarter_retA = quarter_retA.dropna() #删除空值
quarter_accretA = (datasetA["accret_m"]-datasetA["accret_m"].shift(60))/datasetA["accret_m"].shift(60)
quarter_accretA = quarter_accretA.dropna()
quarter_exretA = quarter_retA - quarter_accretA


half_retA = (PGHH["CLOSE"]-PGHH["CLOSE"].shift(125))/PGHH["CLOSE"].shift(125)
half_retA = half_retA.dropna() #删除空值
half_accretA = (datasetA["accret_m"]-datasetA["accret_m"].shift(60))/datasetA["accret_m"].shift(60)
half_accretA = half_accretA.dropna()
half_exretA = half_retA - half_accretA

year_retA = (PGHH["CLOSE"]-PGHH["CLOSE"].shift(250))/PGHH["CLOSE"].shift(250)
year_retA = year_retA.dropna() #删除空值
year_accretA = (datasetA["accret_m"]-datasetA["accret_m"].shift(60))/datasetA["accret_m"].shift(60)
year_accretA = year_accretA.dropna()
year_exretA = year_retA-year_accretA


# In[54]:


res11 = list(filter(lambda x: x > 0, quarter_retA))
quarterwin_ratio1 = len(res11)/len(quarter_exretA)

res1 = list(filter(lambda x: x > 0, quarter_exretA))
quarterwin_ratio = len(res1)/len(quarter_exretA)

res21 = list(filter(lambda x: x > 0, half_retA))
halfwin_ratio1 = len(res21)/len(half_exretA)

res2 = list(filter(lambda x: x > 0, half_exretA))
halfwin_ratio = len(res2)/len(half_exretA)

res31 = list(filter(lambda x: x > 0, year_retA))
yearwin_ratio1 = len(res31)/len(year_exretA)

res3 = list(filter(lambda x: x > 0, year_exretA))
yearwin_ratio = len(res3)/len(year_exretA)



name_dict = {'统计频率':[
                    '季度',
                    '半年度',  
                    '年度']}
df_winratio = pd.DataFrame(name_dict)

df_winratio["超额胜率"] = [quarterwin_ratio,halfwin_ratio,yearwin_ratio]
df_winratio["胜率"] = [quarterwin_ratio1,halfwin_ratio1,yearwin_ratio1]


df_winratio["胜率"] = df_winratio["胜率"].apply(lambda x: format(x, '.2%'))
df_winratio["超额胜率"] = df_winratio["超额胜率"].apply(lambda x: format(x, '.2%'))

df_winratio



# In[55]:


def MaxDrawdown(NV):
    MaxLose = 0
    for i in range(len(NV)):
        output = np.min(NV.iloc[i:])/NV.iloc[i]-1
        if output < MaxLose:
            MaxLose = output
    return abs(MaxLose)
a = datasetA['Fund']

#调用函数，转百分号小数位格式
MDD = "%.4f%%" % (MaxDrawdown(a) * 100)

print("最大回撤率:",MDD)


# # 2.风格分析

# ## 2.1 基于收益率/净值

# In[56]:


dataset20 = w.wsd("399373.SZ,399372.SZ,399377.SZ,399376.SZ", "pct_chg", "2018-03-01", "2021-08-31", "",usedf = True)[1]
dataset20 = dataset20.reset_index()


# In[57]:


dataset20["index"] = pd.to_datetime(dataset20["index"])

dataset20.columns = ["date","col1","col2","col3","col4"]


# ### 2.1.1 敏感性分析

# In[58]:


## 以基金净值的涨跌幅对不同风格的指数的涨跌幅进行分析
##注意：此处并未进行标准化，所以回归结果仅仅代表敏感系数，对于贡献率并没有解释意义。

dataset2 = dataset[["date","Fund_ret"]]
dataset2 = pd.DataFrame(dataset2)

dataset21 = pd.merge(dataset2, dataset20, on = ['date'], how = 'inner')

dataset21.columns = ["date","col0","col1","col2","col3","col4"]

dataset21.fillna(0, inplace=True) #去除空值，要不然后面回归会报错。


# In[59]:


#由于中债总财富指数、货币基金指数均不显著，所以去除两个变量，再次进行回归分析。

lm = ols("col0~col1+col2+col3+col4",dataset21).fit()

print("回归结果-大盘价值指数敏感系数：", "%.4f%%" % (lm.params["col1"]/(lm.params.sum()-lm.params["Intercept"])*100 ))
print("回归结果-大盘成长指数敏感系数：", "%.4f%%" % (lm.params["col2"]/(lm.params.sum()-lm.params["Intercept"])*100 ))
print("回归结果-小盘价值指数敏感系数：", "%.4f%%" % (lm.params["col3"]/(lm.params.sum()-lm.params["Intercept"])*100))
print("回归结果-小盘成长指数敏感系数：", "%.4f%%" % (lm.params["col4"]/(lm.params.sum()-lm.params["Intercept"])*100 ))

lm.summary()

# 整体模型的调整R^2变化不大，但是剩余四个解释变量的显著性上升。认为改结果具有统计学意义。


# In[60]:


pyplt = py.offline.plot
dataz = [go.Bar(
            x=["大盘价值", "大盘成长", "小盘价值","小盘成长"],
            y=[lm.params["col1"]/(lm.params.sum()-lm.params["Intercept"]), lm.params["col2"]/(lm.params.sum()-lm.params["Intercept"]), 
               lm.params["col3"]/(lm.params.sum()-lm.params["Intercept"]),lm.params["col4"]/(lm.params.sum()-lm.params["Intercept"])],
            width = 0.4,
)]
layout = go.Layout(
            title = '基金净值对于各风格指数敏感性系数',
  
)
figure = go.Figure(data = dataz, layout = layout)
iplot(figure)


# In[61]:


dataset21 = dataset21[["date","col0","col1","col2","col3","col4"]]

l = len(dataset21)
date = []
Intercept = []
col1 = []
col2 = []
col3 = []
col4 = []
for i in range(0,l-250+1):
    my_data = dataset21[i:(i+250)]
    my_data = my_data.reset_index(drop = True)
    # 用my_data进行回归，可以考虑用statsmodels模块
    lm = ols("col0~col1+col2+col3+col4",my_data).fit()
    x = my_data.iloc[249,0]
    x0 = lm.params["Intercept"]
    x1 = lm.params["col1"]
    x2 = lm.params["col2"]
    x3 = lm.params["col3"]
    x4 = lm.params["col4"]
    date.append(x)
    Intercept.append(x0)
    col1.append(x1)
    col2.append(x2)
    col3.append(x3)
    col4.append(x4)


result = pd.DataFrame({'date':date,'Intercept':Intercept,'col1':col1,'col2':col2,'col3':col3,'col4':col4})


# In[62]:


#给结果画图

data2 = [
    
    go.Scatter(
        x=result['date'],
        y=result['col1'],
        name = '大盘价值指数敏感系数'),
        
    go.Scatter(
        x=result['date'],
        y=result['col2'],
        name = '大盘成长指数敏感系数'),
    
    go.Scatter(
        x=result['date'],
        y=result['col3'],
        name = '小盘价值指数敏感系数'),
    
    go.Scatter(
        x=result['date'],
        y=result['col4'],
        name = '小盘成长指数敏感系数')
    
]
 
layout = go.Layout(
    title = '基于净值的基金风格分析'
)
 
fig = go.Figure(data = data2,layout=layout)


fig.update_layout(xaxis = dict(tickmode = 'auto',
                              nticks = 10,))

fig.show()


# ### 2.1.2 影响程度分析

# In[63]:


dataset212 = dataset21.copy()
dataset212["col0"] = (dataset212["col0"]-dataset212["col0"].mean())/dataset212["col0"].std()
dataset212["col1"] = (dataset212["col1"]-dataset212["col1"].mean())/dataset212["col1"].std()
dataset212["col2"] = (dataset212["col2"]-dataset212["col2"].mean())/dataset212["col2"].std()
dataset212["col3"] = (dataset212["col3"]-dataset212["col3"].mean())/dataset212["col3"].std()
dataset212["col4"] = (dataset212["col4"]-dataset212["col4"].mean())/dataset212["col4"].std()


# In[64]:


lm2 = ols("col0~col1+col2+col3+col4",dataset212).fit()

print("回归结果-大盘价值指数敏感系数：", "%.4f%%" % (lm2.params["col1"]*100 ))
print("回归结果-大盘成长指数敏感系数：", "%.4f%%" % (lm2.params["col2"]*100 ))
print("回归结果-小盘价值指数敏感系数：", "%.4f%%" % (lm2.params["col3"]*100))
print("回归结果-小盘成长指数敏感系数：", "%.4f%%" % (lm2.params["col4"]*100 ))

print(lm2.summary())


# In[65]:


l = len(dataset212)
date = []
Intercept = []
col1 = []
col2 = []
col3 = []
col4 = []
for i in range(0,l-250+1):
    my_data = dataset21[i:(i+250)]
    my_data["col0"] = (my_data["col0"]-my_data["col0"].mean())/my_data["col0"].std()
    my_data["col1"] = (my_data["col1"]-my_data["col1"].mean())/my_data["col1"].std()
    my_data["col2"] = (my_data["col2"]-my_data["col2"].mean())/my_data["col2"].std()
    my_data["col3"] = (my_data["col3"]-my_data["col3"].mean())/my_data["col3"].std()
    my_data["col4"] = (my_data["col4"]-my_data["col4"].mean())/my_data["col4"].std()
    my_data = my_data.reset_index(drop = True)
    # 用my_data进行回归，可以考虑用statsmodels模块
    lm2 = ols("col0~col1+col2+col3+col4",my_data).fit()
    x = my_data.iloc[249,0]
    x0 = lm2.params["Intercept"]
    x1 = lm2.params["col1"]
    x2 = lm2.params["col2"]
    x3 = lm2.params["col3"]
    x4 = lm2.params["col4"]
    date.append(x)
    Intercept.append(x0)
    col1.append(x1)
    col2.append(x2)
    col3.append(x3)
    col4.append(x4)


result2 = pd.DataFrame({'date':date,'Intercept':Intercept,'col1':col1,'col2':col2,'col3':col3,'col4':col4})


# In[66]:


#给结果画图

data2 = [
    
    go.Scatter(
        x=result2['date'],
        y=result2['col1'],
        name = '大盘价值指数敏感系数'),
        
    go.Scatter(
        x=result2['date'],
        y=result2['col2'],
        name = '大盘成长指数敏感系数'),
    
    go.Scatter(
        x=result2['date'],
        y=result2['col3'],
        name = '小盘价值指数敏感系数'),
    
    go.Scatter(
        x=result2['date'],
        y=result2['col4'],
        name = '小盘成长指数敏感系数')
    
]
 
layout = go.Layout(
    title = '基于净值的基金风格分析'
)
 
fig = go.Figure(data = data2,layout=layout)


fig.update_layout(xaxis = dict(tickmode = 'auto',
                              nticks = 10,))

fig.show()


# In[67]:


result2_abs =  result2.copy().apply(lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x)
result2_abs = result2_abs[["col1","col2","col3","col4"]]
result2_abs_cumsum = result2_abs.cumsum(axis = 1)
result2_abs_cumsum["col1"] =  result2_abs_cumsum["col1"]/ result2_abs_cumsum["col4"]*100
result2_abs_cumsum["col2"] =  result2_abs_cumsum["col2"]/ result2_abs_cumsum["col4"]*100
result2_abs_cumsum["col3"] =  result2_abs_cumsum["col3"]/ result2_abs_cumsum["col4"]*100
result2_abs_cumsum["col4"] =  result2_abs_cumsum["col4"]/ result2_abs_cumsum["col4"]*100


# In[68]:


data_1 = go.Scatter(
        x=result2['date'],
        y=result2_abs_cumsum['col1'],
        name = '大盘价值指数贡献率',
        mode = 'lines',
        line = dict(width=0.5,
              color = 'rgb(135, 206, 250)'),
        fill = 'tonexty'
)
    

data_2 = go.Scatter(
        x=result2['date'],
        y=result2_abs_cumsum['col2'],
    name = '大盘成长指数贡献率',
    mode = 'lines',
    line = dict(width=0.5,
              color = 'rgb(255,255,0)'),
    fill = 'tonexty'
)
 
data_3 = go.Scatter(
        x=result2['date'],
        y=result2_abs_cumsum['col3'],
    name = '小盘价值指数贡献率',
    mode = 'lines',
    line = dict(width=0.5,
              color='rgb(173, 255, 47)'),
    fill='tonexty'
)
 
data_4 = go.Scatter(
        x=result2['date'],
        y=result2_abs_cumsum['col4'],
    name = '小盘成长指数贡献率',
    mode = 'lines',
    line = dict(width=0.5,
              color='rgb(147, 112, 219)'),
    fill='tonexty'
)
 
data = [data_1, data_2, data_3, data_4]
 
layout = go.Layout(
    title = '基金风格贡献比例图',
    showlegend = True,
    xaxis = dict(
        type = 'category',
    ),
    yaxis = dict(
        type = 'linear',
        range = [1, 100],
        dtick = 20,
        ticksuffix = '%'
    )
)
 
pyplt = py.offline.plot

fig = go.Figure(data = data, layout = layout)

fig.update_layout(xaxis = dict(tickmode = 'auto',
                              nticks = 10,))


fig.show()


# ## 2.2 持仓分析

# In[69]:


dataset221 = w.wsd("399373.SZ,399372.SZ,399377.SZ,399376.SZ", "close", "2018-03-01", "2021-08-31", "",usedf = True)[1]
dataset221 = dataset221.reset_index() 


# In[70]:


#导数据

dataset22 = pd.read_csv("cwgs.csv")

dataset22 = dataset22.reset_index(drop = True)
dataset22.columns=['date','cg',"cgbd"]

#转换各列的格式，从str转成float64，要不然后边不好算.
dataset22['cg'] = dataset22['cg'].astype('float64')


# In[71]:


data = [
    
    go.Scatter(
        x=dataset221['index'],
        y=dataset221['399373.SZ'],
        name = '大盘价值指数'),
 
    go.Scatter(
        x=dataset221['index'],
        y=dataset221['399372.SZ'],
        name = '大盘成长指数'),
       
    go.Scatter(
        x=dataset['date'],
        y=dataset['H00906.CSI'],
        name = '中证800全收益指数')
    
 ]


        
layout = go.Layout(
   
)
 
fig = go.Figure(data = data,layout=layout)

#让横坐标只显示10个


iplot(fig)


# In[72]:


data1 = go.Scatter(
        x=dataset22['date'],
        y=dataset22['cg'],
        name = '持股比例',
        mode = 'lines',
        line = dict(width=0.5,
              color = 'rgb(80, 200, 250)'),
    fill = 'tonexty'
)



layout = go.Layout(
    title = '基金风格贡献比例图',
    showlegend = True,
 
    yaxis = dict(
        type = 'linear',
        range = [60, 100],
        dtick = 10,
        ticksuffix = '%'
    )
)
 
pyplt = py.offline.plot

fig = go.Figure(data = data1, layout = layout)

fig.update_layout(xaxis = dict(tickmode = 'auto',
                              nticks = 10,))


fig.show()


       


# In[73]:



trace1 =    go.Scatter(
        x=dataset['date'][18:],
        y=dataset['H00906.CSI'][18:],
        name = '中证800全收益指数')



trace2 = go.Scatter(
        x=dataset22['date'],
        y=dataset22['cg'],
        name = '持股比例',
        mode = 'lines',
        line = dict(width=0.5,
              color = 'rgb(80, 206, 300)'),
    fill = 'tonexty',
     xaxis ='x', 
    yaxis='y2'#标明设置一个不同于trace1的一个坐标轴
)

data = [trace1, trace2]
layout = go.Layout(
     title = '基金股票持仓变动分析',
    showlegend = True,
    yaxis = dict(
        type = 'linear',
        range = [3000, 8000],
        dtick = 1000
    ),
    yaxis2=dict(anchor='x', overlaying='y', side='right', range = [80, 160],
        dtick = 20,
        ticksuffix = '%')#设置坐标轴的格式，一般次坐标轴在右侧
)
 

    
    
    
fig = go.Figure(data=data, layout=layout)
fig.show()


# # 3.选股、择时能力分析

# In[74]:


# step1.导数据
#这边的无风险利率是怎么做出来的，市场利率是怎么做出来的需要确认一下。

data = dataset.copy()


data.fillna(0, inplace=True) #去除空值，要不然后面回归会报错。

data["Ex_Rp"] = data["Fund_ret"]-data["SHIBOR"]
data["Ex_Rm"] = data["ret_m"]-data["SHIBOR"]


# ## 3.1CAPM回归结果

# ![image.png](attachment:image.png)

# In[75]:


y = data[['Ex_Rp']]  # 模型输出变量矩阵

X_CAPM = sm.add_constant(data[['Ex_Rm']])  # CAPM模型输入变量矩阵

CAPM = sm.OLS(y, X_CAPM).fit() # CAPM拟合模型

print(CAPM.summary())

#其中，β 表示组合在市场中暴露的风险敞口；
#α 表示组合无法由市场风险解释的，或者说由个股选择带来的超额收益。
#因此，若某只组合或基金的 α 显著大于0，则意味着基金管理人具有正向的择股能力


# ## 3.2 T-M模型回归结果

# ![image.png](attachment:image.png)

# In[76]:


data['sqr_Ex_Rm'] = data['Ex_Rm']**2

y = data[['Ex_Rp']]  # 模型输出变量矩阵

X_TM = sm.add_constant(data[['Ex_Rm', 'sqr_Ex_Rm']]) # T-M模型输入变量矩阵
TM = sm.OLS(y, X_TM).fit() # T-M拟合模型

print(TM.summary())


#可以看出：若 α 显著大于0，则说明基金管理人具有正向的择券能力；若 β2 显著大于0，则说明基金管理人具有正向的择时能力


# In[77]:


#滞后一项的结果

data32 = data.copy()
data32['sqr_Ex_Rm'] = (data32['Ex_Rm']**2).shift(1)
data32['Ex_Rm'] = data32['Ex_Rm']
data32 = data32.dropna()

y = data32[['Ex_Rp']]  # 模型输出变量矩阵


X_TM = sm.add_constant(data32[['Ex_Rm', 'sqr_Ex_Rm']]) # T-M模型输入变量矩阵
TM = sm.OLS(y, X_TM).fit() # T-M拟合模型

print(TM.summary())


#可以看出：若 α 显著大于0，则说明基金管理人具有正向的择券能力；若 β2 显著大于0，则说明基金管理人具有正向的择时能力


# ## 3.3 H-M、C-L模型分析结果

# ### H-M
# ![image.png](attachment:image.png)

# ### C-L
# ![image.png](attachment:image.png)

# In[78]:


# Step 2: 回归解释变量准备

data['Ex_Rm+'], data['Ex_Rm-'] = data['Ex_Rm'].copy(), data['Ex_Rm'].copy()
data = data.dropna()

for i in data.index:
    if data['Ex_Rm'][i] >= 0:
        data['Ex_Rm+'][i] = data['Ex_Rm'][i]
        data['Ex_Rm-'][i] = 0
    else:
        data['Ex_Rm-'][i] = data['Ex_Rm'][i]
        data['Ex_Rm+'][i] = 0
        

# Step 3: 模型输入-输出变量准备
y = data[['Ex_Rp']]  # 模型输出变量矩阵


X_HM = sm.add_constant(data[['Ex_Rm', 'Ex_Rm+']]) # H-M模型输入变量矩阵
X_CL = sm.add_constant(data[['Ex_Rm-', 'Ex_Rm+']]) # C-L模型输入变量矩阵

# Step 4: 模型训练

HM = sm.OLS(y, X_HM).fit() # H-M拟合模型
CL = sm.OLS(y, X_CL).fit() # C-L拟合模型


# Step 5: 输出结果

print(HM.summary())
print(CL.summary())


print(CL.params)

#C-L模型中的 β2 - β1 实际上与H-M模型的 β2是等价的，这与我们在模型中对择时能力的定义是一致的。因此，从这个角度来说H-M模型和C-L模型是等价的。


# In[79]:


#汇总三个模型的回归结果

result3 = pd.concat([CAPM.params,TM.params,HM.params,CL.params],axis=1,ignore_index=True)
result3.columns =["CAPM","T-M","H-M","C-L"]
result3 = result3.reindex(index = ["const","Ex_Rm","sqr_Ex_Rm","Ex_Rm+","Ex_Rm-"])
result3


# In[ ]:




