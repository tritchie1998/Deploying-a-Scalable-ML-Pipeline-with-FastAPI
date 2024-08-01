import warnings
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.data import process_data
from ml.model import (
    performance_on_categorical_slice
)

warnings.filterwarnings("ignore")


# Load the dataset
file_path = 'data/census.csv'
data = pd.read_csv(file_path)

# Define variables
categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']
label = 'salary'


def test_process_data_training():
    """
    Test the process_data function with complete data in training mode.
    """
    X, y, encoder, lb = process_data(data, categorical_features, label, training=True)
    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)


def test_process_data_missing_categorical():
    """
    Test the process_data function with missing values in categorical features.
    """
    data_missing_cat = data.copy()
    data_missing_cat.loc[0, 'workclass'] = None
    X, y, encoder, lb = process_data(data_missing_cat, categorical_features, label, training=True)
    assert X.shape[0] == data_missing_cat.shape[0]


def test_process_data_missing_continuous():
    """
    Test the process_data function with missing values in continuous features.
    """
    data_missing_cont = data.copy()
    data_missing_cont.loc[0, 'age'] = None
    X, y, encoder, lb = process_data(data_missing_cont, categorical_features, label, training=True)
    assert X.shape[0] == data_missing_cont.shape[0]


def test_process_data_inference():
    """
    Test the process_data function in inference mode.
    """
    X_train, y_train, encoder, lb = process_data(data, categorical_features, label, training=True)
    X_test, y_test, _, _ = process_data(data, categorical_features, label, training=False, encoder=encoder, lb=lb)
    assert X_test.shape == X_train.shape
    assert y_test.shape == y_train.shape


def test_performance_on_categorical_slice_valid():
    """
    Test the performance_on_categorical_slice function for valid slice computation.
    """
    X, y, encoder, lb = process_data(data, categorical_features, label, training=True)
    model = RandomForestClassifier()
    model.fit(X, y)
    precision, recall, fbeta = performance_on_categorical_slice(
        data, column_name='education', slice_value='Bachelors', categorical_features=categorical_features,
        label=label, encoder=encoder, lb=lb, model=model
    )
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_performance_on_categorical_slice_invalid():
    """
    Test the performance_on_categorical_slice function with an invalid slice value.
    """
    X, y, encoder, lb = process_data(data, categorical_features, label, training=True)
    model = RandomForestClassifier()
    model.fit(X, y)
    with pytest.raises(ValueError):
        precision, recall, fbeta = performance_on_categorical_slice(
            data, column_name='education', slice_value='InvalidValue', categorical_features=categorical_features,
            label=label, encoder=encoder, lb=lb, model=model
        )
