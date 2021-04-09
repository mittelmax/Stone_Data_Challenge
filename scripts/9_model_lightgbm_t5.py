from stone_data_challenge.config import data_dir
import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import optuna
import pickle

# # # RUN OPTIONS
TUNING = False
TUNING_SIZE = 200000
TRAINING = False
TRAINING_SIZE = 5000000
FINAL_PREDICTION = False


# # Reading dataframe
print("Reading Dataframe")
df = pd.read_csv(f"{data_dir}/data/feat_eng/model_data_v5.csv")
print("Dataframe succesfully readed")


# # Dropping columns without lag e MA2
df.drop(["caixas_papelao", "caixas_papelao_varpct", "ipca_pct", "cambio_dol", "demissoes", "ind_varejo",
         "TPV_MA_2", "ipca_MA_2", "caixas_papelao_MA_2", "caixas_papelao_varpct_MA_2",
         "demissoes_MA_2", "ind_varejo_MA_2", "cambio_dol_MA_2"], axis="columns", inplace=True)


# # Shifting TPV_mensal four rows for 5 lag prediction (November)
df["TPV_t5"] = df.groupby("id")["TPV_mensal"].shift(-4)
df.drop("TPV_mensal", axis="columns", inplace=True)


# # Using LabelEncoder for categorical variables
# # Lightgbm can handle label encoded variables very efficiently
le = LabelEncoder()
df["MCC"] = le.fit_transform(df["MCC"])
df["MacroClassificacao"] = le.fit_transform(df["MacroClassificacao"])
df["segmento"] = le.fit_transform(df["segmento"])
df["sub_segmento"] = le.fit_transform(df["sub_segmento"])
df["persona"] = le.fit_transform(df["persona"])
df["Estado"] = le.fit_transform(df["Estado"])
df["tipo_documento"] = le.fit_transform(df["tipo_documento"])


# # Saving dataset for prediction
df["mes_referencia"] = pd.to_datetime(df["mes_referencia"])
df_prediction = df.groupby("id").tail(5).set_index("mes_referencia")
df_prediction_december = df_prediction.groupby("id").head(5).groupby("id").tail(1).drop("TPV_t5", axis="columns")
df_prediction_december["ind"] = df_prediction_december["id"]
df_prediction_december = df_prediction_december.set_index("ind")


# # Separating training and test dataset
df = df[df["mes_referencia"] < "2020-08-30"]

df_val = df.groupby("id").tail(5).groupby("id").head(1)
df_val = df_val[pd.notnull(df_val["TPV_t5"])].set_index("mes_referencia")
df_val = df_val.reset_index()
df_val = df_val.sort_values(["id", "mes_referencia"])
df_val = df_val.set_index("mes_referencia")

df_train = df.drop(df.groupby("id").tail(9).index, axis=0).set_index("mes_referencia")
df_train = df_train[pd.notnull(df_train["TPV_t5"])]
df_train = df_train.head(TRAINING_SIZE)

df = df.set_index("mes_referencia")
df = df[pd.notnull(df["TPV_t5"])]


# # Saving column names for later
colnames = df.columns


# # Training hyperparameters
x_colnames = list(df.drop("TPV_t5", axis="columns").columns.values)
categorical_vars = ["id", "MCC", "MacroClassificacao", "segmento", "sub_segmento", "persona", "tipo_documento", "Estado"]


# # Grouped Time Series Split
# Code extracted from link below
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupTimeSeriesSplit
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                           'b', 'b', 'b', 'b', 'b',\
                           'c', 'c', 'c', 'c',\
                           'd', 'd', 'd'])
    >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    ...     print("TRAIN GROUP:", groups[train_idx],\
                  "TEST GROUP:", groups[test_idx])
    TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    TEST GROUP: ['c' 'c' 'c' 'c']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    TEST: [15, 16, 17]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    TEST GROUP: ['d' 'd' 'd']
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]


if TUNING:
    df_train_tuning = df_train.head(TUNING_SIZE)

    # # Objective Function
    def objective(trial, cv_fold_func=np.average):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting': 'rf',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 200),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10)}

        # fitting and returning RMSE scores
        mae_list = []
        for train_idx, test_idx in GroupTimeSeriesSplit().split(df_train_tuning, groups=df_train_tuning["id"]):
            train_x, test_x = df_train_tuning.drop("TPV_t5", axis="columns").values[train_idx], df_train_tuning.drop("TPV_t5", axis="columns").values[test_idx]
            train_y, test_y = df_train_tuning["TPV_t5"][train_idx], df_train_tuning["TPV_t5"][test_idx]

            train_data = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical_vars, feature_name=x_colnames)

            model = lgb.train(params, train_data)
            pred = model.predict(test_x)
            mae = mean_absolute_error(pred, test_y)
            mae_list.append(mae)

        print("Trial done: MAE values on folds: {}".format(mae_list))
        return cv_fold_func(mae_list)


    # # Optuna results
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=80)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)
    best_params = study.best_trial.params

    # Building parameter dictionary
    fixed_params = {'objective': 'regression',
                    'metric': 'mae',
                    'boosting': 'gbdt',
                    'num_threads': 4}

    best_params = {**best_params, **fixed_params}

    # # Saving best params
    filename = f"{data_dir}/data/model/parameters/best_params_t5.pickle"
    outfile = open(filename, "wb")
    pickle.dump(best_params, outfile)
    outfile.close()

else:
    # # Reading model parameters
    infile = open(f"{data_dir}/data/model/parameters/best_params_t5.pickle", "rb")
    best_params = pickle.load(infile)


if TRAINING:
    print("Splitting train and test sets...")
    # # Training model
    for train_idx, test_idx in GroupTimeSeriesSplit().split(df_train, groups=df_train["id"]):
        train_x, test_x = df_train.drop("TPV_t5", axis="columns").values[train_idx], df_train.drop("TPV_t5", axis="columns").values[test_idx]
        train_y, test_y = df_train["TPV_t5"][train_idx], df_train["TPV_t5"][test_idx]
    print("Beginning training...")
    train_data = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical_vars, feature_name=x_colnames)
    test_data = lgb.Dataset(test_x, label=test_y, categorical_feature=categorical_vars, feature_name=x_colnames)
    lgbm_model = lgb.train(best_params, train_data, 250, valid_sets=test_data, early_stopping_rounds=50, verbose_eval=20)

    # # Predicting on Test Data

    # Creating validation sets for each month
    df_val_t5_y = df_val["TPV_t5"]
    df_val_t5_x = df_val.drop(["TPV_t5"], axis="columns")
    y_pred_lgb_test = lgbm_model.predict(df_val_t5_x, num_iteration=lgbm_model.best_iteration)

    # # Model performance on test data
    MAE_lgb_test = mean_absolute_error(y_pred_lgb_test, df_val_t5_y)

    # Feature importances
    imp = pd.DataFrame(lgbm_model.feature_importance()).T
    imp.columns = x_colnames
    imp = imp.T
    imp.columns = ["feat_importance"]
    imp = imp.sort_values("feat_importance")

    # Saving Results
    imp.to_csv(f"{data_dir}/data/model/performance/importances_t5.csv")

    filename = f"{data_dir}/data/model/performance/MAE_t5.pickle"
    outfile = open(filename, "wb")
    pickle.dump(MAE_lgb_test, outfile)
    outfile.close()

    # Joining test set and predictions
    df_y_pred = pd.DataFrame(y_pred_lgb_test, columns=["TPV_pred_t5"])
    df_val_t5_x = df_val_t5_x.reset_index()
    df_val_t5_y = pd.DataFrame(df_val_t5_y)
    df_val_t5_y = df_val_t5_y.reset_index().drop("mes_referencia", axis="columns")
    df_pred_t5 = pd.concat([df_val_t5_x, df_val_t5_y, df_y_pred], axis="columns").set_index("mes_referencia")
    df_pred_t5 = df_pred_t5[["id", "TPV_t5", "TPV_pred_t5"]]

    # Saving dataframe
    df_pred_t5.to_csv(f"{data_dir}/data/performance/test_pred_t5.csv")


if FINAL_PREDICTION:
    print("Training on all data...")
    # # Final Prediction
    print("Splitting train and test sets...")
    for train_idx, test_idx in GroupTimeSeriesSplit().split(df, groups=df["id"]):
        train_x, test_x = df.drop("TPV_t5", axis="columns").values[train_idx], df.drop("TPV_t5", axis="columns").values[test_idx]
        train_y, test_y = df["TPV_t5"][train_idx], df["TPV_t5"][test_idx]
    print("Beginning training...")
    train_data = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical_vars, feature_name=x_colnames)
    test_data = lgb.Dataset(test_x, label=test_y, categorical_feature=categorical_vars, feature_name=x_colnames)
    lgbm_model = lgb.train(best_params, train_data, 250, valid_sets=test_data, early_stopping_rounds=50, verbose_eval=10)

    # # Predicting on all Data
    # Creating validation sets for each month
    y_pred_lgb_final = lgbm_model.predict(df_prediction_december, num_iteration=lgbm_model.best_iteration)

    # # Exporting final results
    y_pred_lgb_final = pd.DataFrame(y_pred_lgb_final, columns=["TPV Dezembro"])
    y_pred_lgb_final.to_csv(f"{data_dir}/data/predictions/prediction_december.csv")
