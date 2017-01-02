from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from scipy.stats import skew
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import math

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


def rmse_cv(model):
	rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
	return rmse


def predict_file(log_price):
	pa = np.expm1(log_price)
	with open('predict.csv', 'w') as f:
		f.write("Id,SalePrice\n")
		for i in range(1459):
			f.write(str(i + 1461) + ',' + str(pa[i]) + '\n')


def plot_model(model):
	p_pred = np.expm1(model.predict(X_train))
	plt.scatter(p_pred, np.expm1(y))
	plt.plot([min(p_pred), max(p_pred)], [min(p_pred), max(p_pred)], c="red")
	plt.show()

def compute_predict(p1, p2):
	a1 = 0.7
	a2 = 0.3
	return p1 * a1 + p2 * a2


def model_predict(all_data, model_type):
	model = ''
	if model_type == 'ridge':
		model_ridge = Ridge()
		alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
		cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
		cv_ridge = pd.Series(cv_ridge, index=alphas)
		print 'Ridge model cv score: ', cv_ridge
		model = Ridge(alpha=10)
		model.fit(X_train, y)
		print model

	elif model_type == 'lasso':
		'''model_Lasso, benchmark cv = 0.110055288106'''
		model = LassoCV(alphas=[1, 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001], max_iter=50000).fit(X_train, y)
		# model_lasso = LassoCV(alphas=[10, 1, 0.1, 0.01], max_iter=10000).fit(X_train, y)
		print "\nmodel Lasso CV: ", rmse_cv(model).mean()
	# coef = pd.Series(model_lasso.coef_, index=X_train.columns)
	# print "\n Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +
	#  str(sum(coef == 0)) + " variables\n"
	# imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
	# print imp_coef

	elif model_type == 'rf':
		model = RandomForestRegressor(10)
		model.fit(X_train, y)
		print 'random forest CV:', rmse_cv(model).mean()

	elif model_type == 'boost':
		'''benchmark xgboost CV:  0.115992295618'''
		model = xgb.XGBRegressor(
			colsample_bytree=0.2,
			gamma=0.0,
			learning_rate=0.005,
			max_depth=4,
			min_child_weight=2,
			n_estimators=10000,
			reg_alpha=0.9,
			reg_lambda=0.6,
			subsample=0.2,
			seed=42,
			silent=True)

		model.fit(X_train, y)
		print 'xgboost CV: ', rmse_cv(model).mean()
	else:
		print '\nNO model selected!!!'
	print model
	plot_model(model)
	return model.predict(X_test)


if __name__ == '__main__':
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
	# print all_data.shape

	# add polynomial GrLivArea features
	# train_data['sq_glArea'] = train_data['GrLivArea'].astype(int).apply(lambda x: math.log(x) ** 2)
	# test_data['sq_glArea'] = test_data['GrLivArea'].astype(int).apply(lambda x: math.log(x) ** 2)
	# train_data['cub_glArea'] = test_data['GrLivArea'].astype(int).apply(lambda x: math.log(x) ** 3)
	# test_data['cub_glArea'] = test_data['GrLivArea'].astype(int).apply(lambda x: math.log(x) ** 3)
	# # this features definitely cause over fitting
	# all_data = pd.concat(
	# 	(train_data.loc[:, 'MSSubClass':'cub_glArea'], test_data.loc[:, 'MSSubClass':'cub_glArea']))


	all_data = pd.concat(
		(train_data.loc[:, 'MSSubClass':'SaleCondition'], test_data.loc[:, 'MSSubClass':'SaleCondition']))
	all_data["house_age"] = 2010 - all_data["YearBuilt"]
	all_data["TimeSinceSold"] = 2010 - all_data["YrSold"]
	all_data["YearsSinceRemodel"] = all_data["YrSold"] - all_data["YearRemodAdd"]
	# all_data.drop('YearBuilt', axis=1, inplace=True)
	# all_data.drop('YrSold', axis=1, inplace=True)
	# all_data.drop('YearRemodAdd', axis=1, inplace=True)

	# all_data['porchArea'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + \
	# 						all_data['3SsnPorch'] + all_data['ScreenPorch']
	# # all_data.drop('OpenPorchSF', axis=1, inplace=True)
	# # all_data.drop('EnclosedPorch', axis=1, inplace=True)
	# # all_data.drop('3SsnPorch', axis=1, inplace=True)
	# # all_data.drop('ScreenPorch', axis=1, inplace=True)

	# neighborhood_map = {
	# 	"MeadowV": 0,  # 88000
	# 	"IDOTRR": 1,  # 103000
	# 	"BrDale": 1,  # 106000
	# 	"OldTown": 1,  # 119000
	# 	"Edwards": 1,  # 119500
	# 	"BrkSide": 1,  # 124300
	# 	"Sawyer": 1,  # 135000
	# 	"Blueste": 1,  # 137500
	# 	"SWISU": 2,  # 139500
	# 	"NAmes": 2,  # 140000
	# 	"NPkVill": 2,  # 146000
	# 	"Mitchel": 2,  # 153500
	# 	"SawyerW": 2,  # 179900
	# 	"Gilbert": 2,  # 181000
	# 	"NWAmes": 2,  # 182900
	# 	"Blmngtn": 2,  # 191000
	# 	"CollgCr": 2,  # 197200
	# 	"ClearCr": 3,  # 200250
	# 	"Crawfor": 3,  # 200624
	# 	"Veenker": 3,  # 218000
	# 	"Somerst": 3,  # 225500
	# 	"Timber": 3,  # 228475
	# 	"StoneBr": 4,  # 278000
	# 	"NoRidge": 4,  # 290000
	# 	"NridgHt": 4,  # 315000
	# }
	#
	# all_data["NeighborhoodBin"] = all_data["Neighborhood"].map(neighborhood_map)

	# convert quality(str) to numerical
	# qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
	# qual_dict = {None: 0, "Po": 0.5, "Fa": 1, "TA": 1.5, "Gd": 2, "Ex": 2.5}
	# all_data["ExterQual"] = all_data["ExterQual"].map(qual_dict).astype(int)
	# all_data["ExterCond"] = all_data["ExterCond"].map(qual_dict).astype(int)
	# all_data["BsmtQual"] = all_data["BsmtQual"].map(qual_dict).astype(int)
	# all_data["BsmtCond"] = all_data["BsmtCond"].map(qual_dict).astype(int)
	# all_data["HeatingQC"] = all_data["HeatingQC"].map(qual_dict).astype(int)
	# all_data["KitchenQual"] = all_data["KitchenQual"].map(qual_dict).astype(int)
	# all_data["FireplaceQu"] = all_data["FireplaceQu"].map(qual_dict).astype(int)
	# all_data["GarageQual"] = all_data["GarageQual"].map(qual_dict).astype(int)
	# all_data["GarageCond"] = all_data["GarageCond"].map(qual_dict).astype(int)


	# drop seems useless features, result: not work well
	# all_data = all_data.drop('RoofMatl', axis=1)  # only one is not zero
	# all_data = all_data.drop('Condition2', axis=1)  # only two is not zero
	# all_data = all_data.drop('MSZoning', axis=1)
	# all_data = all_data.drop('MSSubClass', axis=1)

	# log transform the target:
	train_data["SalePrice"] = np.log1p(train_price)

	# log transform skewed numeric features:
	numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
	skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna().astype(float)))  # compute skewness
	skewed_feats = skewed_feats[skewed_feats > 0.7]
	# print "skew_features: "
	# print skewed_feats
	skewed_feats = skewed_feats.index
	all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

	# filling NA's with 0 and mean respectively:
	all_data["GarageYrBlt"].fillna(0.0, inplace=True)
	all_data = all_data.fillna(all_data.mean())
	# filling NA's with majority value
	# all_data = all_data.fillna(all_data.mode(numeric_only=False).iloc[0])

	# Convert categorical variable into dummy/indicator variables
	all_data = pd.get_dummies(all_data)
	print 'all_data shape: ', all_data.shape

	X_train = all_data[:train_data.shape[0]]  # all_data[:1460]
	X_test = all_data[train_data.shape[0]:]
	y = train_data.SalePrice

	# predict_value = model_predict(all_data, 'lasso')
	predict_value = compute_predict(model_predict(all_data, 'lasso'), model_predict(all_data, 'boost'))
	predict_file(predict_value)
