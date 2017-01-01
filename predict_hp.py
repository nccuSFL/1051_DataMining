from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from scipy.stats import skew
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


def rmse_cv(model):
	rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
	return rmse


def predict_file(log_price):
	pa = np.expm1(log_price)
	with open('predict.csv', 'w') as f:
		f.write("Id,SalePrice\n")
		for i in range(1459):
			f.write(str(i+1461) + ',' + str(pa[i]) + '\n')


if __name__ == '__main__':
	"""load data, train model, calculate cost """
	train_data = pd.read_csv("train.csv")
	test_data = pd.read_csv("test.csv")

	# pre_processing data

	# *** remove outlier, GrLivArea > 4500, cv = 0.110 ***
	# small_set = train_data[train_data["GrLivArea"] > 4000]
	# print small_set["GrLivArea"]
	# print small_set["SalePrice"]
	train_data.drop(train_data[train_data["GrLivArea"] > 4500].index, inplace=True)
	# train_data.drop(train_data[train_data["GrLivArea"] == 4676].index, inplace = True)
	# train_data.drop(train_data[train_data["GrLivArea"] == 5642].index, inplace=True)

	# add new features
	train_price = train_data['SalePrice']
	train_data.drop('SalePrice', axis=1)

	# add sqrt GrLivArea features
	# train_data['sqrt_grlivArea'] = train_data['GrLivArea'].astype(int).apply(math.sqrt)
	# test_data['sqrt_grlivArea'] = test_data['GrLivArea'].astype(int).apply(math.sqrt)
	# all_data = pd.concat(
	# 	(train_data.loc[:, 'MSSubClass':'sqrt_grlivArea'], test_data.loc[:, 'MSSubClass':'sqrt_grlivArea']))


	# add polynomial GrLivArea features
	# train_data['sq_glArea'] = train_data['GrLivArea'].astype(int).apply(lambda x: x ** 2)
	# test_data['sq_glArea'] = test_data['GrLivArea'].astype(int).apply(lambda x: x ** 2)
	# train_data['cub_glArea'] = test_data['GrLivArea'].astype(int).apply(lambda x: x ** 3)
	# test_data['cub_glArea'] = test_data['GrLivArea'].astype(int).apply(lambda x: x ** 3)
	# # this features definitely couse overfitting
	# all_data = pd.concat(
	# 	(train_data.loc[:, 'MSSubClass':'cub_glArea'], test_data.loc[:, 'MSSubClass':'cub_glArea']))

	all_data = pd.concat(
		(train_data.loc[:, 'MSSubClass':'SaleCondition'], test_data.loc[:, 'MSSubClass':'SaleCondition']))

	# drop seems useless features, result: not work well
	# all_data = all_data.drop('RoofMatl', axis=1)  # only one is not zero
	# all_data = all_data.drop('Condition2', axis=1)  # only two is not zero
	# all_data = all_data.drop('MSZoning', axis=1)
	# all_data = all_data.drop('MSSubClass', axis=1)

	# log transform the target:

	train_data["SalePrice"] = np.log1p(train_price)

	# log transform skewed numeric features:
	numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
	skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
	# print "skew_features: "
	# print skewed_feats
	skewed_feats = skewed_feats[skewed_feats > 0.75]
	skewed_feats = skewed_feats.index
	all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

	# filling NA's with the mean of the column:
	all_data = all_data.fillna(all_data.mean())

	# Convert categorical variable into dummy/indicator variables
	all_data = pd.get_dummies(all_data)

	X_train = all_data[:train_data.shape[0]]  # all_data[:1460]
	X_test = all_data[train_data.shape[0]:]
	y = train_data.SalePrice

	# linear regression: model ridge
	model_ridge = Ridge()
	alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
	cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
	cv_ridge = pd.Series(cv_ridge, index=alphas)
	model_ridge = Ridge(alpha=10)
	model_ridge.fit(X_train, y)
	print model_ridge
	predict_file(model_ridge.predict(X_test))
	print 'Ridge model cv score: '
	print cv_ridge

	# model_Lasso, benchmark cv = 0.11007685145
	# model_lasso = LassoCV(alphas=[10, 1, 0.1, 0.01, 0.001, 0.0005]).fit(X_train, y)
	# model_lasso = LassoCV(alphas=[10, 1, 0.1, 0.01], max_iter=10000).fit(X_train, y)
	# predict_file(model_lasso.predict(X_test))
	# print model_lasso
	# print "\nmodel Lasso CV: "
	# print rmse_cv(model_lasso).mean()

	# coef = pd.Series(model_lasso.coef_, index=X_train.columns)
	# print "\n Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables\n"
	# imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
	# print imp_coef


