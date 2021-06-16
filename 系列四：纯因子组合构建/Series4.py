import numpy as np
import pandas as pd
import os
import sys
sys.path.append("C:\\Users\\yyhan\\Desktop\\自定义包\\FactorAnalysis\\")
from FactorAnalysis import *
from empyrical import cum_returns, annual_return, annual_volatility, sharpe_ratio
import statsmodels.api as sm

# 导入数据并整合
data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
factor_names = [x for x in os.listdir(data_path) if "." not in x]  # 所有大类因子名
factor_names.remove("industry")

def comp(fac_name):
    """
    输入大类因子名，返回等权合成后的因子暴露

    Parameters
    ----------
    fac_name: str
        大类因子名.
    
    Returns
    -------
    等权合成后的因子暴露：dict of pd.Series
    """
    factor_path = os.path.join(data_path, fac_name)
    factor = read_data(factor_path, **{"index_col": 0})
    Compeq(factor, list(range(len(list(factor.values())[0].columns))), name=fac_name)

    return dict(zip(factor.keys(), [x.iloc[:, -1] for x in factor.values()]))

comp_factor = dict(zip(factor_names, [comp(x) for x in factor_names]))
all_months = list(comp_factor.values())[0]  # 所有月份

def merge_factor():
    """
    将各风格因子合并到一个表格
    """
    result = {}
    for mon in all_months:
        result[mon] = pd.concat(
            [comp_factor[name][mon] for name in factor_names],
            axis=1,
            join="inner",
        )

    return result

style_factor = merge_factor()  # 合并后的风格因子

# 行业因子与风格因子合并并加入收益率、市值
industry_factor = read_data(
    os.path.join(data_path, "industry"),
    **{"index_col": 0, "encoding": "gbk"}
)
factors = add_data(industry_factor, style_factor)
r_t1 = pd.read_csv(os.path.join(data_path, "r_t+1.csv"), index_col=0)
cap = pd.read_csv(os.path.join(data_path, "MktCap.csv"), index_col=0)
all_data = add_data(factors, r_t1, "r_t+1")
all_data = add_data(all_data, cap, "cap")

# 风格因子去缺失值、去极值、标准化处理
def PreRaw():
    result = {}
    for mon in all_months:
        temp = all_data[mon].copy()
        temp = temp.dropna()
        temp.iloc[:, 29:36] = temp.iloc[:, 29:36].apply(lambda x: OutProcess(x, method="AM"))
        temp.iloc[:, 29:36] = temp.iloc[:, 29:36].apply(lambda x: Standardization(
            x, 
            method="CWM",
            cap=temp.iloc[:, 37]
        ))
        result[mon] = temp

    return result

all_data_pro = PreRaw()

industry_names = list(list(all_data_pro.values())[0].columns[:29])
# WLS求解因子收益
def WLS():
    """
    WLS求解纯因子收益

    Returns
    -------
    因子收益：pd.DataFrame
        index为日期.
    因子收益t值：pd.DataFrame
        index为日期.
    R方：pd.Series
        每期截面回归的R方.
    """
    fret, tvalues = [], []
    R2 = pd.Series([np.nan] * len(all_months), index=all_months)
    for mon in all_months:
        temp = all_data_pro[mon].copy()
        model = sm.WLS(
            temp.iloc[:, 36].values,
            sm.add_constant(temp.iloc[:, :36].values),
            weights=temp.iloc[:, -1].values
        ).fit()  # 拟合
        fret.append(model.params)
        tvalues.append(model.tvalues)
        R2.loc[mon] = model.rsquared
    
    fret = pd.DataFrame(fret, index=all_months, columns=industry_names + factor_names)
    tvalues = pd.DataFrame(tvalues, index=all_months, columns=industry_names + factor_names)

    return fret, tvalues, R2

fret1, tvalues1, WLSR2 = WLS()

# 保存结果
with pd.ExcelWriter("result.xlsx") as writer:
    fret1.to_excel(writer, "纯因子收益（WLS）")
    tvalues1.to_excel(writer, "WLS t")

with pd.ExcelFile("result.xlsx") as reader:
    fret1 = pd.read_excel(reader, "纯因子收益（WLS）", index_col=0)
    tvalues1 = pd.read_excel(reader, "WLS t", index_col=0)

# 计算股票利用率
def stock_use_ratio():
    ratio = pd.Series([np.nan] * len(all_months), index=all_months)
    for mon in all_months:
        ratio.loc[mon] = len(all_data_pro[mon]) / len(all_data[mon])

    return ratio

stock_ratio = stock_use_ratio()

# 纯行业因子相关统计
def summary1():
    ret = fret1.iloc[:, 1:30]     # 纯行业因子收益
    t = tvalues1.iloc[:, 1:30]    # 纯行业因子t值
    result = []
    result.append(t.apply(lambda x: (x.abs() > 2).sum() / len(x)))          # 因子显著度
    result.append(t.apply(lambda x: (x.abs().mean())))                      # t值绝对值平均
    result.append(ret.apply(lambda x: annual_return(x, period="monthly")))  # 因子年化收益
    result.append(ret.apply(annual_volatility, **{"period": "monthly"}))    # 因子年化波动
    result.append(ret.apply(lambda x: sharpe_ratio(x, period="monthly")))   # 因子收益夏普比率

    result = pd.concat(
        result,
        axis=1,
        keys=[
            "因子显著度", "t值绝对值平均", "因子年化收益",
            "因子年化波动", "因子收益率夏普"
        ]
    )
    result.iloc[:, [0, 2, 3, 4]] = result.iloc[:, [0, 2, 3, 4]].applymap(lambda x: "%.2f%%" % (x * 100))

    return result

summary1()

# 纯风格因子相关统计
def auto_stable(fac_name):
    """
    计算某个因子自稳定相关系数的时序均值
    
    Parameters
    ----------
    fac_name: str
        因子名.
    
    Returns
    -------
    自稳定相关系数均值：float
    """
    result = []
    for i in range(len(all_months) - 1):
        temp = pd.concat(
            [
                all_data_pro[all_months[i]][fac_name], 
                all_data_pro[all_months[i+1]][fac_name],
                all_data_pro[all_months[i]]["cap"]
            ],
            axis=1,
            join="inner",
            keys=["t", "t+1", "cap"]
        ).dropna()
        temp["weight"] = temp["cap"] / temp["cap"].sum()
        D1 = temp["t"] - temp["t"].mean()
        D2 = temp["t+1"] - temp["t+1"].mean()
        tho = (temp["weight"] * D1 * D2).sum() / np.sqrt((temp["weight"] * D1**2).sum() * (temp["weight"] * D2**2).sum())
        result.append(tho)

def summary2():
    ret = fret1.iloc[:, 30:37]     # 纯风格因子收益
    t = tvalues1.iloc[:, 30:37]    # 纯风格因子t值
    result = []
    result.append(t.apply(lambda x: (x.abs() > 2).sum() / len(x)))          # 因子显著度
    result.append(t.apply(lambda x: (x.abs().mean())))                      # t值绝对值平均
    result.append(ret.apply(lambda x: annual_return(x, period="monthly")))  # 因子年化收益
    result.append(ret.apply(annual_volatility, **{"period": "monthly"}))    # 因子年化波动
    result.append(ret.apply(lambda x: sharpe_ratio(x, period="monthly")))   # 因子收益夏普比率

    result = pd.concat(
        result,
        axis=1,
        keys=[
            "因子显著度", "t值绝对值平均", "因子年化收益",
            "因子年化波动", "因子收益率夏普"
        ]
    )
    result.iloc[:, [0, 2, 3, 4]] = result.iloc[:, [0, 2, 3, 4]].applymap(lambda x: "%.2f%%" % (x * 100))

    return result