#!/usr/bin/env python3
"""
def integrate(func, start, stop):
    # uses trapezoid instead of QUADPACK
    from scipy.integrate import trapezoid
    from numpy import arange, vectorize

    step = 1e-3

    func = vectorize(func)

    x = arange(start, stop, step)
    y = func(x)

    print(f'x={x}, y={y}')
    return trapezoid(y, x)
"""

def integrate(func, start, stop):
    from scipy.integrate import quadrature

    val, _ = quadrature(func, start, stop, maxiter=5000, miniter=5000)

    return val


def prepare_dataframe():
    filenames = [
        ('data/S1/S1.csv', 'Demand'),
        ('data/S2/S2.csv', 'Demand'),
        ('data/S3/AT.csv', 'Demand'),
        ('data/S3/BE.csv', 'Demand'),
        ('data/S3/BG.csv', 'Demand'),
        ('data/S3/CH.csv', 'Demand'),
        ('data/S3/CZ.csv', 'Demand'),
        ('data/S3/DK.csv', 'Demand'),
        ('data/S3/ES.csv', 'Demand'),
        ('data/S3/FR.csv', 'Demand'),
        ('data/S3/GR.csv', 'Demand'),
        ('data/S3/IT.csv', 'Demand'),
        ('data/S3/NL.csv', 'Demand'),
        ('data/S3/PT.csv', 'Demand'),
        ('data/S3/SI.csv', 'Demand'),
        ('data/S3/SK.csv', 'Demand'),
        ('data/S4/MIDATL.csv', 'Net'),
        ('data/S4/SOUTH.csv', 'Net'),
        ('data/S4/WEST.csv', 'Net'),
    ]
    from pandas import read_csv, concat, DataFrame, merge
    dataframes = [
        # (dataframe, zone)
        (
            read_csv(filename).rename(columns={rename_col: 'y'})[['Hour', 'y']],
            filename
        )
        for filename, rename_col in filenames
    ]

    # add zone name
    for dataframe, zone in dataframes:
        dataframe['Zone'] = zone

    dataframes, _ = zip(*dataframes)

    # concat vertically
    df = concat(dataframes, ignore_index=True)

    df_group = df.groupby(['Zone', 'Hour'])

    # randomly select 1 sample from each group
    df = df_group.apply(DataFrame.sample, n=1).reset_index(drop=True)

    # hourly mean and std per zone
    mu = df_group.mean().rename(columns={'y': 'mu'})
    sigma = df_group.std().rename(columns={'y': 'sigma'})

    # concat horizontally
    df = merge(df, mu, on=['Zone', 'Hour'])
    df = merge(df, sigma, on=['Zone', 'Hour'])

    df.index.name = 'Sample ID'

    return df

def numeric_crps(y, mu, sigma):
    from scipy.stats import norm

    def crps_integral_term1(x, mu, sigma):
        # calculates F²
        z = (x - mu) / sigma
        F = norm.cdf(z)
        return F**2

    def crps_integral_term2(x, mu, sigma):
        # calculates (1-F)²
        z = (x - mu) / sigma
        F = norm.cdf(z)
        return (1-F)**2

    almost_inf = sigma * 100

    term1 = integrate(
        lambda x: crps_integral_term1(x, mu, sigma), -almost_inf, y)

    term2 = integrate(
        lambda x: crps_integral_term2(x, mu, sigma), y, almost_inf)

    return term1 + term2

def numeric_crps_ord_dmu(s):
    delta = 1e-6

    mu = s['mu']
    sigma = s['sigma']
    y = s['y']

    num = numeric_crps(y, mu + delta, sigma) - s['num_crps']
    den = delta

    return num / den

def numeric_crps_ord_dlogsigma(s):
    delta = 1e-6

    mu = s['mu']
    sigma = s['sigma']
    y = s['y']

    from numpy import log
    num = numeric_crps(y, mu, sigma + delta) - s['num_crps']
    den = log(sigma + delta) - log(sigma)

    return num / den

def add_symbolic_results(df):
    from scipy.stats import norm
    from numpy import sqrt, pi

    z = (df['y'] - df['mu']) / df['sigma']
    sigma = df['sigma']

    df['sym_crps'] = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / sqrt(pi)
    )

    df['sym_ord_dmu'] = 1 - 2 * norm.cdf(z)
    df['sym_ord_dlogsigma'] = sigma * (2 * norm.pdf(z) - 1 / sqrt(pi))

    m11 = 1 / sigma / sqrt(pi)
    m12 = 0
    m22 = sigma / 2 / sqrt(pi)

    df['sym_nat_dmu'] = m11 * df['sym_ord_dmu'] + m12 * df['sym_ord_dlogsigma']
    df['sym_nat_dlogsigma'] = m12 * df['sym_ord_dmu'] + m22 * df['sym_ord_dlogsigma']

    df['sym_m11'] = m11
    df['sym_m12'] = m12
    df['sym_m22'] = m22

    return df


def numeric_M11(x, mu, sigma):
    from scipy.stats import norm
    z = (x - mu) / sigma
    return 2 * norm.pdf(z)**2 / sigma**2

def numeric_M12(x, mu, sigma):
    from scipy.stats import norm
    z = (x - mu) / sigma
    return 2 * norm.pdf(z)**2 * (x - mu) / sigma**2

def numeric_M22(x, mu, sigma):
    from scipy.stats import norm
    z = (x - mu) / sigma
    return 2 * norm.pdf(z)**2 * (z**2)

def numeric_natural_gradient(s):
    mu = s['mu']
    sigma = s['sigma']

    almost_inf = sigma * 100

    m11 = integrate(
        lambda x: numeric_M11(x, mu, sigma), -almost_inf, almost_inf)

    m12 = integrate(
        lambda x: numeric_M12(x, mu, sigma), -almost_inf, almost_inf)

    m22 = integrate(
        lambda x: numeric_M22(x, mu, sigma), -almost_inf, almost_inf)

    s['num_nat_dmu'] = m11 * s['num_ord_dmu'] + m12 * s['num_ord_dlogsigma']
    s['num_nat_dlogsigma'] = m12 * s['num_ord_dmu'] + m22 * s['num_ord_dlogsigma']

    s['num_m11'] = m11
    s['num_m12'] = m12
    s['num_m22'] = m22

    return s

def add_error_columns(df):
    terms = ['crps', 'ord_dmu', 'ord_dlogsigma', 'nat_dmu', 'nat_dlogsigma',
             'm11', 'm12', 'm22']
    for term in terms:
        df[f'err_{term}'] = (df[f'num_{term}'] - df[f'sym_{term}']).abs()

    for term in terms:
        df[f'err_{term}%'] = df[f'err_{term}'] / df[f'sym_{term}'].abs() * 100

    return df


from diskcache import Cache
cache = Cache('cache')


def run_all():
    df = prepare_dataframe()
    df = add_symbolic_results(df)
    df['num_crps'] = df.apply(lambda s: numeric_crps(s['y'], s['mu'], s['sigma']), axis=1)
    df['num_ord_dmu'] = df.apply(numeric_crps_ord_dmu, axis=1)
    df['num_ord_dlogsigma'] = df.apply(numeric_crps_ord_dlogsigma, axis=1)
    df = df.apply(numeric_natural_gradient, axis=1, result_type='expand')
    df = add_error_columns(df)
    df.to_csv('verify-zonal-hourly.csv')
    return df

df = run_all()
from pandas import set_option
set_option('display.max_columns', None)
print(df)
from matplotlib.pyplot import scatter, show
scatter(df['err_m11'], df['err_m22'])
show()
