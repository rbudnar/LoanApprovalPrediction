XGBOOST_PARAMETERS = [
    {
        "name": "xgb_learning_rate",
        "type": "range",
        "bounds": [0.001, 0.5],
        "log_scale": True,
    },
    {
        "name": "xgb_n_estimators",
        "type": "range",
        "bounds": [50, 3000],
        "value_type": "int",
    },
    {
        "name": "xgb_max_depth",
        "type": "range",
        "bounds": [3, 30],
        "value_type": "int",
    },
    {
        "name": "xgb_min_child_weight",
        "type": "range",
        "bounds": [0.1, 1],
        "value_type": "float",
    },
    {
        "name": "xgb_subsample",
        "type": "range",
        "bounds": [0.1, 1.0],
    },
    {
        "name": "xgb_colsample_bytree",
        "type": "range",
        "bounds": [0.1, 1.0],
    },
    {
        "name": "xgb_gamma",
        "type": "range",
        "bounds": [0, 5],
    },
]
LIGHTGBM_PARAMETERS = [
    {
        "name": "lgb_learning_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": True,
    },
    {
        "name": "lgb_n_estimators",
        "type": "range",
        "bounds": [50, 3000],
        "value_type": "int",
    },
    {
        "name": "lgb_max_depth",
        "type": "range",
        "bounds": [15, 30],
        "value_type": "int",
    },
    {
        "name": "lgb_num_leaves",
        "type": "range",
        "bounds": [10, 100],
        "value_type": "int",
    },
    {
        "name": "lgb_min_child_samples",
        "type": "range",
        "bounds": [10, 100],
        "value_type": "int",
    },
    {
        "name": "lgb_subsample",
        "type": "range",
        "bounds": [0.1, 1.0],
    },
    {
        "name": "lgb_colsample_bytree",
        "type": "range",
        "bounds": [0.1, 1.0],
    },
    {
        "name": "lgb_reg_alpha",
        "type": "range",
        "bounds": [0, 2],
    },
    {
        "name": "lgb_reg_lambda",
        "type": "range",
        "bounds": [0, 2],
    },
]
CATBOOST_PARAMETERS = [
    {
        "name": "cat_learning_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": True,
    },
    {
        "name": "cat_iterations",
        "type": "range",
        "bounds": [50, 3000],
        "value_type": "int",
    },
    {
        "name": "cat_depth",
        "type": "range",
        "bounds": [3, 16],
        "value_type": "int",
    },
    {
        "name": "cat_l2_leaf_reg",
        "type": "range",
        "bounds": [1, 10],
    },
    {
        "name": "cat_bagging_temperature",
        "type": "range",
        "bounds": [0, 100],
    },
]


def get_xgb_params(parameters):
    return {
        "learning_rate": parameters["xgb_learning_rate"],
        "n_estimators": parameters["xgb_n_estimators"],
        "max_depth": parameters["xgb_max_depth"],
        "min_child_weight": parameters["xgb_min_child_weight"],
        "subsample": parameters["xgb_subsample"],
        "colsample_bytree": parameters["xgb_colsample_bytree"],
        "gamma": parameters["xgb_gamma"],
        "enable_categorical": True,
    }


def get_lgb_params(parameters):
    return {
        "learning_rate": parameters["lgb_learning_rate"],
        "n_estimators": parameters["lgb_n_estimators"],
        "max_depth": parameters["lgb_max_depth"],
        "num_leaves": parameters["lgb_num_leaves"],
        "min_child_samples": parameters["lgb_min_child_samples"],
        "subsample": parameters["lgb_subsample"],
        "colsample_bytree": parameters["lgb_colsample_bytree"],
        "reg_alpha": parameters["lgb_reg_alpha"],
        "reg_lambda": parameters["lgb_reg_lambda"],
    }


def get_cat_params(parameters, categorical_variables: list[str]):
    return {
        "learning_rate": parameters["cat_learning_rate"],
        "iterations": parameters["cat_iterations"],
        "depth": parameters["cat_depth"],
        "l2_leaf_reg": parameters["cat_l2_leaf_reg"],
        "bagging_temperature": parameters["cat_bagging_temperature"],
        "cat_features": categorical_variables,
    }
