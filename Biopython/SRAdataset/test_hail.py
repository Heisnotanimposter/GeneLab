import hail as hl
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

def initialize_hail():
    hl.init(log='/tmp/hail_log/hail.log')

def load_preprocessed_data():
    # Load the preprocessed Hail Table
    ht = hl.read_table('output/preprocessed.ht')
    df = ht.to_pandas()
    return df

def test_model(df):
    # Load the trained model
    model = joblib.load('output/trained_model.pkl')

    # Extract features and target variable
    X = pd.json_normalize(df['receptor_features'])
    y = df['smiles'].apply(lambda x: len(x))  # Example target: length of SMILES string

    # Predict and evaluate
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')

def main():
    initialize_hail()
    df = load_preprocessed_data()
    test_model(df)

main()