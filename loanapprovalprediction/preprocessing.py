from pathlib import Path
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class ProcessedDataset(BaseModel):
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    X_test: pd.DataFrame
    scaler: StandardScaler

    class Config:
        arbitrary_types_allowed = True

    def __iter__(self):
        return iter(
            (
                self.X_train,
                self.X_valid,
                self.y_train,
                self.y_valid,
                self.X_test,
                self.scaler,
            )
        )


def preprocess_data(
    data_: pd.DataFrame,
    scaler: StandardScaler,
    cont_features: list[str],
    cat_features: list[str],
    fit: bool = False,
) -> pd.DataFrame:
    data = data_.reset_index(drop=True)
    if fit:
        X_cont = scaler.fit_transform(data[cont_features])
    else:
        X_cont = scaler.transform(data[cont_features])

    return pd.concat(
        [
            pd.DataFrame(X_cont, columns=cont_features),
            data[cat_features].apply(pd.Categorical),
        ],
        axis=1,
    )


def prepare_dataset(
    data_dir: Path,
    cont_features: list[str],
    cat_features: list[str],
    target: str,
    use_original_dataset: bool = False,
) -> ProcessedDataset:
    original_dataset = pd.read_csv(data_dir / "credit_risk_dataset.csv")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    scaler = StandardScaler()

    if use_original_dataset:
        dataset = pd.concat(
            [original_dataset, train.drop(columns=["id"])], ignore_index=True
        ).assign(id=lambda x: x.index)
    else:
        dataset = train

    training, valid = train_test_split(dataset, test_size=0.2, random_state=42)
    X_train = preprocess_data(
        training,
        scaler,
        cont_features=cont_features,
        cat_features=cat_features,
        fit=True,
    )
    X_valid = preprocess_data(
        valid,
        scaler,
        cont_features=cont_features,
        cat_features=cat_features,
    )
    y_train = training[target]
    y_valid = valid[target]

    X_test = preprocess_data(
        test,
        scaler,
        cont_features=cont_features,
        cat_features=cat_features,
    )
    return ProcessedDataset(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        X_test=X_test,
        scaler=scaler,
    )
