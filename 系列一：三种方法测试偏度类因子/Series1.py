import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_1samp
import empyrical as em
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\yyhan\\Desktop\\自定义包\\FactorAnalysis\\")
from FactorAnalysis import *

data_path = os.path.join(os.path.dirname(os.getcwd()), "data")  # data path
skew_path = os.path.join(data_path, "Skewness")  # skewness data path
ind_path = os.path.join(data_path, "industry")  # industry factor path

# load data
r_t1 = pd.read_csv(os.path.join(data_path, "r_t+1.csv"), index_col=0)
cap = pd.read_csv(os.path.join(data_path, "MktCap.csv"), index_col=0)
ind = read_data(ind_path, index_col=0, encoding="gbk")
skew = read_data(skew_path, index_col=0)
factor_names = skew["2009-11-30"].columns  # names of all factors

############ Test with WLS ##############
# preprocessing
def PreRaw():
    """
    Preprocessing of skewness factors.

    Returns
    -------
    factors after treated: dict.
    """
    result = {}
    for mon in skew.keys():
        temp = skew[mon]
        temp = temp.apply(lambda x: OutProcess(x, method="AM"))
        temp = temp.apply(lambda x: Standardization(x, method="EWM"))
        temp = temp.fillna(0)
        result[mon] = temp
    
    return result

skew1 = PreRaw()  # processed skewness factors for WLS

def DicttoDf():
    """
    Transform skewness dict to df.
    
    Returns
    -------
    Isolated skewness factors as df: list of pd.DataFrame
    """
    result = []
    for f in factor_names:
        result.append(
            pd.concat(
                [skew1[mon][f] for mon in r_t1.columns],
                axis=1,
                keys=r_t1.columns,
                join="outer",
                sort=True
            )
        )
    
    return result

skew1df = DicttoDf()  # transform to list of df

WLS_result = [FactorRegs(r_t1, ind, i, cap) for i in skew1df]

def SummaryWLS():
    """
    Summary results of WLS.

    Returns
    -------
    summary results: pd.DataFrame
        Columns are factor names.
    """
    indi1 = [x[1].abs().mean() for x in WLS_result]
    indi2 = [(x[1].abs() > 2).sum() / len(x[1]) for x in WLS_result]
    indi3 = [x[0].mean() for x in WLS_result]
    indi4 = [ttest_1samp(x[0], 0)[0] for x in WLS_result]
    indi5 = [x[1].mean() / x[1].std() for x in WLS_result]

    # to df
    temp = pd.DataFrame(
        [indi1, indi2, indi3, indi4, indi5],
        index=["tabs均值", "tabs大于2占比", "因子收益率均值", "因子收益率序列t检验", "t值序列均值/标准差"],
        columns=factor_names
    ).T

    return temp

SummaryWLS().to_clipboard()  # statistic result

def plot_cumr():
    """
    Plot cumulative returns of all factors.
    """
    temp = pd.DataFrame(
        [em.cum_returns(x[0]) for x in WLS_result],
        index=factor_names,
        columns=r_t1.columns
    ).T
    temp.plot(figsize=(10, 8))

plot_cumr()

############# Test with IC ############
# add rt+1, cap, ind to skew1
def AddtoSkew1():
    temp = add_data(skew1, ind)  # add industry factor
    temp = add_data(temp, r_t1, "r_t+1")  # add r_t+1
    temp = add_data(temp, cap, "cap")  # add market cap
    
    return temp

skew2 = AddtoSkew1()

# neutralization
def Neutralize():
    for mon in skew2.keys():
        temp = skew2[mon]
        temp.dropna(inplace=True)
        temp.iloc[:, :9] = temp.iloc[:, :9].apply(
            lambda x: Neutralization(x, temp.iloc[:, list(range(9, 38)) + [39]])
        )

Neutralize()

# IC
all_ic = [IC_together(skew2, i, 38) for i in range(9)]

# summary IC results
def SummaryIC():
    result = []
    for ic in all_ic:
        result_i = []
        result_i.append(ic.mean())
        result_i.append(ic.std())
        result_i.append(ic.mean() / ic.std())
        result_i.append((ic > 0).sum() / len(ic))
        result.append(result_i)
    result = pd.DataFrame(
        result,
        index=factor_names,
        columns=["IC序列均值", "IC序列标准差", "IR比率", "IC>0占比"]
    )

    return result

SummaryIC().to_clipboard()

pd.DataFrame(all_ic, index=factor_names, columns=skew2.keys()).T.cumsum().plot()  # cumulative ic