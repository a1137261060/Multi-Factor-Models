import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_1samp
import empyrical as em
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\yyhan\\Desktop\\自定义包\\FactorAnalysis\\")
from FactorAnalysis import *
from empyrical import annual_return, sharpe_ratio, max_drawdown

data_path = os.path.join(os.path.dirname(os.getcwd()), "data")  # data path
skew_path = os.path.join(data_path, "Skewness")  # skewness data path
ind_path = os.path.join(data_path, "industry")  # industry factor path

# load data
r_t1 = pd.read_csv(os.path.join(data_path, "r_t+1.csv"), index_col=0)
cap = pd.read_csv(os.path.join(data_path, "MktCap.csv"), index_col=0)
ind = read_data(ind_path, index_col=0, encoding="gbk")
skew = read_data(skew_path, index_col=0)
factor_names = skew["2009-11-30"].columns  # names of all factors
all_mons = list(skew.keys())  # all months

# preprocess factors
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

skew1 = PreRaw()

# mean Spearman corr
def MeanCorr():
    temp = []
    for mon in all_mons:
        temp.append(skew1[mon].corr(method="spearman"))
    
    return sum(temp) / len(temp)

MeanCorr().to_clipboard()

# add rt+1, cap, ind to skew1
def AddtoSkew1():
    temp = add_data(skew1, ind)  # add industry factor
    temp = add_data(temp, r_t1, "r_t+1")  # add r_t+1
    temp = add_data(temp, cap, "cap")  # add market cap
    
    return temp

skew2 = AddtoSkew1()
skew2[all_mons[10]].head().columns



# compound factors
## equal weighted
Compeq(skew2, list(range(9)))
weight_eq = pd.DataFrame(
    [np.array([1] * len(factor_names)) / len(factor_names)] * len(all_mons),
    index=all_mons,
    columns=factor_names
)
## factor return weighted
### construct log of cap with outlier (not) processed
def lncap():
    for mon in all_mons:
        temp = skew2[mon]
        temp["lncap"] = np.log(temp["cap"])
        temp["lncap_treat"] = OutProcess(temp["lncap"], method="AM")

lncap()
### calculate factor return
factor_return = pd.DataFrame(
    [WLSReg(skew2, list(range(9, 38)) + [41], i, 38, 39)[0] for i in range(9)],
    index=factor_names
).T
### compounded factors
weight_ret = CompW(
    skew2, 
    list(range(9)), 
    factor_return,
    "fac_ret",
)
weight_ret_half = CompW(
    skew2, 
    list(range(9)), 
    factor_return,
    "fac_ret_half",
    ishalf=True
)
## factor IC weighted
### cap and industry neutralization
def Neutralize():
    result = dict()
    for mon in all_mons:
        temp = skew2[mon].dropna().copy()
        temp.iloc[:, :9] = temp.iloc[:, :9].apply(
            lambda x: Neutralization(x, temp.iloc[:, list(range(9, 38)) + [42]])
        )
        result[mon] = temp

    return result

skew2_neu = Neutralize()  # need save this result addtionaly
### calculate factor IC
factor_ic = pd.DataFrame(
    [IC_together(skew2_neu, i, 38) for i in range(9)],
    index=factor_names
).T
### compounded factors
weight_ic=CompW(
    skew2, 
    list(range(9)), 
    factor_ic,
    "fac_ic",
)
weight_ic_half = CompW(
    skew2, 
    list(range(9)), 
    factor_ic,
    "fac_ic_half",
    ishalf=True
)
## max IC_IR
weight_maxicir_samp = CompMax(
    skew2,
    list(range(9)),
    factor_ic,
    "fac_maxicir_samp",
    cov_="sample"
)
weight_maxicir1 = CompMax(
    skew2,
    list(range(9)),
    factor_ic,
    "fac_maxicir1",
)
## max IC
weight_maxic = CompMax(
    skew2,
    list(range(9)),
    factor_ic,
    "fac_maxic",
    max_="IC"
)
## PCA
weight_pca1 = CompPCA(skew2, list(range(9)))



# test and compare factors
## WLS test
all_factors = list(range(9)) + [40] + list(range(43, 51))
comp_factors = [40] + list(range(43, 51))
all_names = skew2[all_mons[0]].iloc[:, all_factors].columns

## preprocess
def PreRaw1():
    result = {}
    for mon in all_mons:
        temp = skew2[mon].copy()
        temp.iloc[:, comp_factors] = temp.iloc[:, comp_factors].apply(lambda x: OutProcess(x, method="AM"))
        temp.iloc[:, comp_factors] = temp.iloc[:, comp_factors].apply(lambda x: Standardization(x, method="EWM"))
        temp.iloc[:, comp_factors] = temp.iloc[:, comp_factors].fillna(0)
        result[mon] = temp
    
    return result

skew3 = PreRaw1()

def WLS_test():
    result = [WLSReg(skew3, list(range(9, 38)) + [41], i, 38, 39) for i in all_factors]
    factor_r = pd.DataFrame([x[0] for x in result], index=all_names).T
    t = pd.DataFrame([x[1] for x in result], index=all_names).T
    temp = pd.DataFrame(
        [
            t.abs().mean(),
            t.apply(lambda x: (x.abs() > 2).sum() / len(x)),
            t.mean(),
            factor_r.mean()
        ],
        index=["abs(t)均值", "abs(t)>2占比", "t均值", "因子收益率均值"]
    ).T

    return temp, factor_r

WLS_result, factor_Ret = WLS_test()


## IC test
### neutralize the compounded factors
def Neutralization_all():
    result = {}
    for mon in all_mons:
        temp = skew3[mon].dropna().copy()
        temp.iloc[:, all_factors] = temp.iloc[:, all_factors].apply(
            lambda x: Neutralization(x, temp.iloc[:, list(range(9, 38)) + [42]])
        )
        result[mon] = temp

    return result

skew3_neu = Neutralization_all() 

def IC_test():
    factor_IC = pd.DataFrame([IC_together(skew3_neu, i, 38) for i in all_factors], index=all_names).T
    result = pd.DataFrame(
        [
            factor_IC.mean(),
            factor_IC.std(),
            factor_IC.mean() / factor_IC.std(),
            factor_IC.apply(lambda x: (x > 0).sum() / len(x))
        ],
        index=["Rank IC均值", "Rank IC标准差", "IC_IR", "IC>0占比"]
    ).T

    return result, factor_IC

IC_result, factor_Ic = IC_test()

pd.concat([WLS_result, IC_result], axis=1).to_clipboard()


## sort portfolio
def sort_test():
    sort_port = [SortP(skew3_neu, i, 38) for i in all_factors]
    result = []
    for sort_p in sort_port:
        result_i = sort_p.apply(annual_return, **{"period": "monthly"}).tolist()
        ls = sort_p["Q5"] - sort_p["Q1"]  # long short portfolio
        result_i.append(annual_return(ls, period="monthly"))
        result_i.append(sharpe_ratio(ls, period="monthly"))
        result_i.append(max_drawdown(ls))
        result_i.append((ls > 0).sum() / len(ls))
        result.append(result_i)
    result = pd.DataFrame(
        result,
        index=all_names,
        columns=list(sort_p.columns) + [
            "多空组合年化收益率",
            "多空组合夏普比率",
            "多空组合最大回撤",
            "多空组合月胜率"
        ]
    )

    return sort_port, result

sort_ports, sort_result = sort_test()

sort_result.to_clipboard()



# for fac_ic
## NAV relative to benchmark
benchmark = pd.read_csv("index.csv", index_col=0)
sort_ic = sort_ports[12]  # sort portfolio of fac_ic
ic_ic = factor_Ic["fac_ic"]  # factor IC of fac_ic
return_ic = factor_Ret["fac_ic"]  # factor return of fac_ic
sort_ic.apply(
    lambda x: pd.Series((1 + x).cumprod().values / (1 + benchmark.iloc[:, 0] / 100).cumprod().values, index=benchmark.index)
).to_clipboard()

## cumulative Rank IC and factor return
pd.DataFrame({
    "累积RankIC": ic_ic.cumsum(),
    "累积因子收益率": return_ic.cumsum()
}).to_clipboard()



# weights analysis
weight_ic.to_clipboard()