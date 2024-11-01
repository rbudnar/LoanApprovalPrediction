XGBOOST_PARAMETERS = [
    {
        "name": "xgb_learning_rate",
        "type": "range",
        "bounds": [0.01, 0.3],
        "log_scale": True,
    },
    {
        "name": "xgb_n_estimators",
        "type": "range",
        "bounds": [50, 300],
        "value_type": "int",
    },
    {
        "name": "xgb_max_depth",
        "type": "range",
        "bounds": [3, 10],
        "value_type": "int",
    },
    {
        "name": "xgb_subsample",
        "type": "range",
        "bounds": [0.5, 1.0],
        "value_type": "float",
    },
    {
        "name": "xgb_colsample_bytree",
        "type": "range",
        "bounds": [0.5, 1.0],
        "value_type": "float",
    },
    {
        "name": "xgb_gamma",
        "type": "range",
        "bounds": [0, 5],
        "value_type": "float",
    },
]

LIGHTGBM_PARAMETERS = [
    {
        "name": "lgb_n_estimators",
        "type": "range",
        "bounds": [50, 300],
        "value_type": "int",
    },
    {
        "name": "lgb_max_depth",
        "type": "range",
        "bounds": [3, 10],
        "value_type": "int",
    },
    {
        "name": "lgb_learning_rate",
        "type": "range",
        "bounds": [0.01, 0.3],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "lgb_num_leaves",
        "type": "range",
        "bounds": [20, 150],
        "value_type": "int",
    },
    {
        "name": "lgb_feature_fraction",
        "type": "range",
        "bounds": [0.5, 1.0],
        "value_type": "float",
    },
    {
        "name": "lgb_bagging_fraction",
        "type": "range",
        "bounds": [0.5, 1.0],
        "value_type": "float",
    },
]
# CATBOOST_PARAMETERS = [
#     {
#         "name": "cat_learning_rate",
#         "type": "range",
#         "bounds": [0.01, 0.5],
#         "log_scale": True,
#     },
#     {
#         "name": "cat_iterations",
#         "type": "range",
#         "bounds": [50, 3000],
#         "value_type": "int",
#     },
#     {
#         "name": "cat_depth",
#         "type": "range",
#         "bounds": [3, 16],
#         "value_type": "int",
#     },
#     {
#         "name": "cat_l2_leaf_reg",
#         "type": "range",
#         "bounds": [1, 10],
#     },
#     {
#         "name": "cat_bagging_temperature",
#         "type": "range",
#         "bounds": [0, 100],
#     },
# ]

CATBOOST_PARAMETERS = [
    {
        "name": "cat_iterations",
        "type": "range",
        "bounds": [50, 300],
        "value_type": "int",
    },
    {
        "name": "cat_depth",
        "type": "range",
        "bounds": [3, 10],
        "value_type": "int",
    },
    {
        "name": "cat_learning_rate",
        "type": "range",
        "bounds": [0.01, 0.3],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "cat_l2_leaf_reg",
        "type": "range",
        "bounds": [1, 10],
        "value_type": "float",
    },
    {
        "name": "cat_bagging_temperature",
        "type": "range",
        "bounds": [0, 1],
        "value_type": "float",
    },
]

LOGISTIC_REGRESSION_PARAMETERS = [
    {
        "name": "lr_C",
        "type": "range",
        "bounds": [0.01, 10],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "lr_max_iter",
        "type": "range",
        "bounds": [100, 1000],
        "value_type": "int",
    },
    {
        "name": "lr_solver",
        "type": "choice",
        "values": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    # {
    #     "name": "lr_penalty",
    #     "type": "choice",
    #     "values": ["l1", "l2", "none"],  # "elasticnet",
    # },
    {
        "name": "lr_l1_ratio",
        "type": "range",
        "bounds": [0, 1],
        "value_type": "float",
    },
]


def get_xgb_params(parameters):
    return {
        "n_estimators": parameters["xgb_n_estimators"],
        "max_depth": parameters["xgb_max_depth"],
        "learning_rate": parameters["xgb_learning_rate"],
        "subsample": parameters["xgb_subsample"],
        "colsample_bytree": parameters["xgb_colsample_bytree"],
        "gamma": parameters["xgb_gamma"],
        "enable_categorical": True,
    }


def get_lgb_params(parameters):
    return {
        "n_estimators": parameters["lgb_n_estimators"],
        "max_depth": parameters["lgb_max_depth"],
        "learning_rate": parameters["lgb_learning_rate"],
        "num_leaves": parameters["lgb_num_leaves"],
        "feature_fraction": parameters["lgb_feature_fraction"],
        "bagging_fraction": parameters["lgb_bagging_fraction"],
    }


def get_cat_params(parameters, categorical_variables: list[str]):
    return {
        "iterations": parameters["cat_iterations"],
        "depth": parameters["cat_depth"],
        "learning_rate": parameters["cat_learning_rate"],
        "l2_leaf_reg": parameters["cat_l2_leaf_reg"],
        "bagging_temperature": parameters["cat_bagging_temperature"],
        "cat_features": categorical_variables,
    }


def get_lr_params(parameters):
    return {
        "C": parameters["lr_C"],
        "max_iter": parameters["lr_max_iter"],
        "solver": parameters["lr_solver"],
        # "penalty": parameters["lr_penalty"],
        "l1_ratio": parameters.get("lr_l1_ratio", None),  # Optional parameter
    }
