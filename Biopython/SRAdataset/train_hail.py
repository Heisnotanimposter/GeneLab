import hail as hl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def initialize_hail():
    hl.init(log='/tmp/hail_log/hail.log')

def load_preprocessed_data():
    # Load the preprocessed Hail Table
    ht = hl.read_table('output/preprocessed.ht')
    df = ht.to_pandas()
    return df

def train_model(df):
    # Example: Train a RandomForestClassifier on the data
    # Assuming df has features in 'receptor_features' and target in 'ligand_smiles'
    
    # Extract features and target variable
    X = pd.json_normalize(df['receptor_features'])
    y = df['smiles'].apply(lambda x: len(x))  # Example target: length of SMILES string

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Validate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy:.2f}')
    
    # Save the trained model
    joblib.dump(model, 'output/trained_model.pkl')

def main():
    initialize_hail()
    df = load_preprocessed_data()
    train_model(df)

main()