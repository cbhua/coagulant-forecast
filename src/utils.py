from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from math import sqrt


def cal_accu(tar, out):
    ''' Calculate RMSE, R2, and Correlation '''
    mse = mean_squared_error(out, tar)
    rmse = sqrt(mse)
    r2 = r2_score(out, tar)
    corr, _ = pearsonr(out, tar)
    return rmse, r2, corr
