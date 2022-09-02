import os
import tarfile

import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import toolbox

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from six.moves import urllib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    csv_path = housing_path + "/housing.csv"
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = load_housing_data(housing_path=HOUSING_PATH)

    # housing.head()
    # housing.info()
    # housing["ocean_proximity"].value_counts()
    # housing.describe()
    # housing.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # # change continues values to numeric labels
    housing["house_value_cat"] = pd.cut(x=housing["median_house_value"], bins=5, labels=[1, 2, 3, 4, 5])

    # # create train and test sets
    train_set, test_set = train_test_split(housing, test_size=0.2, stratify=housing["house_value_cat"], random_state=0)
    # train_set, val_set = train_test_split(train_set, test_size=0.2,
    #                                       stratify=train_set["house_value_cat"], random_state=0)

    # # delete added column
    del housing["house_value_cat"]
    del train_set["house_value_cat"]
    del test_set["house_value_cat"]
    # del val_set["house_value_cat"]

    # train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", c="median_house_value",
                 cmap=plt.get_cmap("jet"), colorbar=True, figsize=(10, 6))
    # plt.show()

    # # calculate correlation
    # corr_matrix = housing.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))

    attributes = ["median_house_value", "median_income", "total_rooms", "latitude"]
    scatter_matrix(housing[attributes], figsize=(10, 8))
    # plt.show()

    # # create new attributes
    # housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    # housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    # housing["population_per_household"] = housing["population"] / housing["households"]
    attr_adder = toolbox.CombinedAttributesAdder(add_bedrooms_per_room=True)
    housing_extra_attribs = attr_adder.transform(housing.values)

    housing.sort_values("households").plot(kind="scatter", x="longitude", y="latitude",
                                           alpha=1, c="households", cmap=plt.get_cmap("jet"),
                                           colorbar=True, figsize=(10, 6))
    # plt.show()

    # # calculate correlation with new attributes
    corr_matrix = housing.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))
    housing.plot(kind="scatter", x="median_house_value", y="total_bedrooms", alpha=0.1)
    # plt.show()

    # # prepare data
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    # # # clean data (fill missing attributes with median)
    # housing_numeric = housing.drop("ocean_proximity", axis=1)
    # imputer = SimpleImputer(strategy="median")
    # imputer.fit(housing_numeric)
    # X = imputer.transform(housing_numeric)
    # # X = imputer.fit_transform(housing_numeric)
    # housing_numeric = pd.DataFrame(X, columns=housing_numeric.columns)
    #
    # # # Handling Text and Categorical Attributes
    # housing_cat = housing[["ocean_proximity"]]
    #
    # # ML algorithms will assume that two nearby values are more similar than two distant values
    # # fine in some cases for ordered categories such as “bad”, “average”, “good”, “excellent”
    # # ordinal_encoder = OrdinalEncoder()
    # # housing_cat_ordinal = ordinal_encoder.fit_transform(housing_cat)
    #
    # # fine for non ordered categories
    # one_hot_encoder = OneHotEncoder()
    # housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)
    #
    # # # Feature Scaling
    # # normalization
    # # scaler_normal = MinMaxScaler()
    # # housing_normal = scaler_normal.fit_transform(housing_numeric)
    #
    # # standardization
    # scaler_standard = StandardScaler()
    # housing_standard = scaler_standard.fit_transform(housing_numeric)

    # Transformation Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", toolbox.CombinedAttributesAdder(add_bedrooms_per_room=True)),
        ("std_scaler", StandardScaler())
    ])
    housing_numeric = housing.drop("ocean_proximity", axis=1)
    # housing_pipe = num_pipeline.fit_transform(housing_numeric)

    num_attribs = list(housing_numeric)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])
    housing_prepared = full_pipeline.fit_transform(housing)

    train_enable = 0
    if train_enable:
        # # training and evaluating
        # linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        housing_predict = lin_reg.predict(housing_prepared)

        # some_data = housing.iloc[:5]
        # some_label = housing_labels.iloc[:5]
        # some_data_p = full_pipeline.transform(some_data)
        # pred = lin_reg.predict(some_data_p)
        # print(f"lin_reg_dif: {abs(pred-some_label)}\n")
        # print("lin_reg_dif: ", abs(housing_predict-housing_labels))

        # RMSE
        lin_mse = mean_squared_error(housing_labels, housing_predict)
        lin_rmse = np.sqrt(lin_mse)
        print(f"linear regression rmse error: {lin_rmse}\n")

        # decision tree regressor
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)
        housing_predict = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predict)
        tree_rmse = np.sqrt(tree_mse)
        print(f"decision tree rmse error: {tree_rmse}\n")
        # K-fold cross validation
        tree_reg_kfold = cross_val_score(tree_reg, X=housing_prepared, y=housing_labels,
                                         scoring="neg_mean_squared_error", cv=10)
        tree_reg_kfold_rmse = np.sqrt(-tree_reg_kfold)
        print(f"decision tree k-fold rmse error: {tree_reg_kfold_rmse}\n"
              f"decision tree k-fold rmse mean: {tree_reg_kfold_rmse.mean()}\n"
              f"decision tree k-fold rmse std: {tree_reg_kfold_rmse.std()}\n")

        # random forests: trains many Decision Trees on random subsets of the features
        # then averaging out their predictions
        forest_reg = RandomForestRegressor()
        forest_reg.fit(housing_prepared, housing_labels)
        housing_predict = forest_reg.predict(housing_prepared)
        forest_mse = mean_squared_error(housing_labels, housing_predict)
        forest_rmse = np.sqrt(forest_mse)
        print(f"random forests rmse error: {forest_rmse}\n")
        # K-fold cross validation
        forest_reg_kfold = cross_val_score(forest_reg, X=housing_prepared, y=housing_labels,
                                           scoring="neg_mean_squared_error", cv=10)
        forest_reg_kfold_rmse = np.sqrt(-forest_reg_kfold)
        print(f"random forests k-fold rmse error: {forest_reg_kfold_rmse}\n"
              f"random forests k-fold rmse mean: {forest_reg_kfold_rmse.mean()}\n"
              f"random forests k-fold rmse std: {forest_reg_kfold_rmse.std()}\n")

    # # save models
    # joblib.dump(lin_reg, "lin_reg.joblib")
    # joblib.dump(tree_reg_kfold, "tree_reg_kfold.joblib")
    # joblib.dump(forest_reg_kfold, "forest_reg_kfold.joblib")
    # # load models
    lin_reg = joblib.load("lin_reg.joblib")
    tree_reg_kfold = joblib.load("tree_reg_kfold.joblib")
    forest_reg_kfold = joblib.load("forest_reg_kfold.joblib")

    grid_enable = 1
    if grid_enable:
        # # Grid Search
        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)
        print(grid_search.best_params_)
        print(grid_search.best_estimator_)
        grid_result = grid_search.cv_results_
        for mean_score, params in zip(grid_result["mean_test_score"], grid_result["params"]):
            print(np.sqrt(-mean_score), params)

        # relative importance of each attribute
        feature_importances = grid_search.best_estimator_.feature_importances_
        extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
        cat_encoder = full_pipeline.named_transformers_["cat"]
        cat_one_hot_attribs = list(cat_encoder.categories_[0])
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        for i in sorted(zip(feature_importances, attributes), reverse=True):
            print(i)

    rand_enable = 0
    if rand_enable:
        # # Randomized Search
        param_rand = {
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=500, num=5)],
            "max_features": [1, 'sqrt'],
            "max_depth": [int(x) for x in np.linspace(5, 30, num=6)],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10]
        }
        random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_rand,
                                           cv=5, n_iter=10, n_jobs=-1, random_state=0,
                                           scoring='neg_mean_squared_error')
        random_search.fit(housing_prepared, housing_labels)
        rand_result = random_search.cv_results_
        for mean_score, params in zip(rand_result["mean_test_score"], rand_result["params"]):
            print(np.sqrt(-mean_score), params)

    # # Evaluate System on the Test Set
    final_model = grid_search.best_estimator_
    x_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    x_test_prepared = full_pipeline.transform(x_test)
    final_predictions = final_model.predict(x_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"final_rmse: {final_rmse}")

    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                   loc=squared_errors.mean(),
                                   scale=stats.sem(squared_errors))))
