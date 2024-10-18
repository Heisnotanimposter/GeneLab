
# Step 1: Install Necessary Packages

# Install required libraries
!pip install -q biopython stable-baselines3 gymnasium shimmy pyspark

# Step 2: Import Libraries

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Bio import Entrez, SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from google.colab import drive
import datetime
import joblib
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession

# Step 3: Mount Google Drive

drive.mount('/content/drive')

# Define the path to your dataset
data_dir = '/content/drive/MyDrive/GeneLab/DNAsequential/'  # Update this path based on your Drive structure

# Verify that the files exist by listing the directory
print("Listing files in the dataset directory:")
!ls {data_dir}

# Define individual file paths
human_file = os.path.join(data_dir, 'human.txt')
dog_file = os.path.join(data_dir, 'dog.txt')
chimpanzee_file = os.path.join(data_dir, 'chimpanzee.txt')
example_dna_file = os.path.join(data_dir, 'example_dna.fa')

# Verify file existence
for file in [human_file, dog_file, chimpanzee_file, example_dna_file]:
    if not os.path.exists(file):
        print(f'File {file} not found. Please check the path.')
    else:
        print(f'File {file} found.')

# Step 4: Define Classes and Functions

# Callback for TensorBoard
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

# Genetic Data Fetcher Class
class GeneticDataFetcher:
    def __init__(self, email):
        self._email = email
        Entrez.email = email

    def fetch_sequence(self, accession):
        try:
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            return str(record.seq)
        except Exception as e:
            print(f"Error fetching sequence: {e}")
            return None

# Sequence Analyzer Class
class SequenceAnalyzer:
    @staticmethod
    def analyze_sequence(sequence, mutation_positions, expected_bases):
        results = []
        for pos in mutation_positions:
            if pos < 0 or pos >= len(sequence):
                results.append(f"Position {pos} is out of range.")
                continue
            actual_base = sequence[pos]
            expected_base = expected_bases.get(pos, None)
            if expected_base and actual_base != expected_base:
                results.append(f"Mutation at {pos}: expected {expected_base}, found {actual_base}")
            elif expected_base:
                results.append(f"No mutation at position {pos}.")
        return results

# Treatment Predictor Class
class TreatmentPredictor:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        self.tensorboard = self._setup_tensorboard()

    def _build_model(self, input_shape):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def _setup_tensorboard(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return tensorboard_callback

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_data=(X_val, y_val),
                       callbacks=[self.tensorboard])

    def evaluate_model(self, X_test, y_test):
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        return accuracy, precision, recall, f1

    def predict_success_probability(self, new_patient_data):
        return self.model.predict(new_patient_data)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = keras.models.load_model(path)

# Genetic Mutation Class
class GeneticMutation:
    def __init__(self, mutation_rate=0.01, elitism_rate=0.1, crossover_rate=0.7):
        self.base_mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.bases = ['A', 'T', 'C', 'G']

    def mutate_sequence(self, sequence, variable_mutation_rate=None):
        sequence = list(sequence)
        for i in range(len(sequence)):
            current_mutation_rate = self.base_mutation_rate
            if variable_mutation_rate:
                current_mutation_rate += variable_mutation_rate.get(i, 0)
            if np.random.rand() < current_mutation_rate:
                original_base = sequence[i]
                new_bases = self.bases.copy()
                new_bases.remove(original_base)
                sequence[i] = np.random.choice(new_bases)
        return ''.join(sequence)

    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1, parent2
        point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def select_elites(self, population, scores, elite_size):
        sorted_indices = np.argsort(scores)[::-1]
        elites = [population[i] for i in sorted_indices[:elite_size]]
        return elites

# Reinforcement Learning Environment
class GeneMutationEnv(gym.Env):
    def __init__(self, initial_sequence, expected_bases, mutation_positions, predictor, mutation_operator):
        super(GeneMutationEnv, self).__init__()
        self.initial_sequence = list(initial_sequence)
        self.sequence = list(initial_sequence)
        self.expected_bases = expected_bases  # Dict of position: expected_base
        self.mutation_positions = mutation_positions  # List of positions to mutate
        self.predictor = predictor
        self.mutation_operator = mutation_operator
        self.steps = 0

        # Define action and observation space
        self.action_space = spaces.MultiBinary(len(self.mutation_positions))
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(len(self.mutation_positions),), dtype=np.float32)

    def reset(self):
        self.sequence = list(self.initial_sequence)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        state = []
        for pos in self.mutation_positions:
            if pos < 0 or pos >= len(self.sequence):
                state.append(0)
            else:
                state.append(1 if self.sequence[pos] == self.expected_bases.get(pos, self.sequence[pos]) else 0)
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        done = False
        reward = 0

        # Apply mutations based on action
        for idx, mutate in enumerate(action):
            if mutate:
                pos = self.mutation_positions[idx]
                if pos < 0 or pos >= len(self.sequence):
                    continue
                original_base = self.sequence[pos]
                new_bases = self.mutation_operator.bases.copy()
                new_bases.remove(original_base)
                self.sequence[pos] = np.random.choice(new_bases)

        # Evaluate the sequence
        correct_mutations = 0
        total_mutations = len(self.expected_bases)
        for pos, expected_base in self.expected_bases.items():
            if pos < 0 or pos >= len(self.sequence):
                continue
            if self.sequence[pos] == expected_base:
                correct_mutations += 1

        # Reward structure
        if correct_mutations == total_mutations:
            reward = 10
            done = True
        else:
            reward = correct_mutations
            if self.steps >= 20:
                done = True

        # Get next state
        next_state = self._get_state()

        return next_state, reward, done, {}

    def render(self, mode='human'):
        print(f"Current Sequence: {''.join(self.sequence)}")

# Data Processing Functions
def get_k_mers(x, kmers):
    '''Divide sequences into words'''
    kmer = ''
    for i in range(len(x)-kmers+1):
        kmer = kmer + str(x[i:i+kmers]) + ' '
    return kmer.strip()

def generate_training_data(x, kmers):
    '''Generate sentences from sequences'''
    new_X = []
    for i in range(len(x)):
        new_X.append(get_k_mers(x[i], kmers))
    new_X = np.array(new_X)
    return new_X

def vectorize_features(x, ngram_range=(4, 4)):
    '''
    Generate a dictionary of sentences having
    made up of 4 words as default.
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer(ngram_range=ngram_range, analyzer='word')
    X_transf = tf.fit_transform(x)
    return X_transf, tf

# Pipeline Function
def pipeline(df, model):
    # get features and labels from the dataframe
    sequences = df['sequence']
    labels = df['class']

    # create new features using kmers and vectorizer
    new_X = generate_training_data(sequences, kmers=6)

    X_transf, tf_vectorizer = vectorize_features(new_X, ngram_range=(4, 4))

    # split the data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X_transf, labels,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=labels)

    print('Number of training samples', X_train.shape[0])
    print('Number of test samples', X_test.shape[0])
    print('Number of features', X_train.shape[1])

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(confusion_matrix(y_test, y_pred_test))

    return y_pred_test, model

# Main Application Class
class MainApplication:
    def __init__(self, email, accession, mutation_positions, expected_bases, data_path):
        self.fetcher = GeneticDataFetcher(email)
        self.analyzer = SequenceAnalyzer()
        self.mutation_operator = GeneticMutation()
        self.accession = accession
        self.mutation_positions = mutation_positions
        self.expected_bases = expected_bases
        self.data_path = data_path

        # Fetch initial sequence
        self.sequence = self.fetcher.fetch_sequence(accession)
        if not self.sequence:
            raise ValueError("Failed to fetch genetic sequence.")

        # Initialize Predictor
        self.predictor = TreatmentPredictor(input_shape=(10,))  # Adjust input shape as needed

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path, sep='\t')  # Assuming tab-separated
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def train_rf_model(self, df):
        # Train a Random Forest model using train_hail.py logic
        # Assuming 'receptor_features' and 'smiles' are present
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
        os.makedirs('output', exist_ok=True)
        joblib.dump(model, 'output/trained_model.pkl')
        print("Random Forest model trained and saved as 'output/trained_model.pkl'")

    def test_ml_model(self, df):
        # Test the model using test_hail.py logic
        # Load the trained model
        model_path = 'output/trained_model.pkl'
        if not os.path.exists(model_path):
            print(f"Trained model not found at {model_path}. Please train the model first.")
            return

        model = joblib.load(model_path)

        # Extract features and target variable
        X = pd.json_normalize(df['receptor_features'])
        y = df['smiles'].apply(lambda x: len(x))  # Example target: length of SMILES string

        # Predict and evaluate
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f'Test Accuracy: {accuracy:.2f}')

    def run(self):
        # Analyze initial sequence
        analysis_results = self.analyzer.analyze_sequence(self.sequence, self.mutation_positions, self.expected_bases)
        for result in analysis_results:
            print(result)

        # Load and prepare data
        df_human = self.load_data()
        if df_human is None:
            print("Data loading failed. Exiting.")
            return

        # Prepare data for ML
        new_X = generate_training_data(df_human['sequence'], kmers=6)
        X_transf, tf_vectorizer = vectorize_features(new_X, ngram_range=(4, 4))

        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(X_transf, df_human['class'],
                                                            test_size=0.3, random_state=42, stratify=df_human['class'])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=0.5, random_state=42, stratify=y_temp)

        # Train the model
        self.predictor.train_model(X_train.toarray(), y_train, X_val.toarray(), y_val, epochs=50, batch_size=32)

        # Evaluate the model
        accuracy, precision, recall, f1 = self.predictor.evaluate_model(X_test.toarray(), y_test)
        print(f"Model Evaluation:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}")

        # Train Random Forest model
        self.train_rf_model(df_human)

        # Test the ML model
        self.test_ml_model(df_human)

        # Initialize RL Environment
        env = GeneMutationEnv(
            initial_sequence=self.sequence,
            expected_bases=self.expected_bases,
            mutation_positions=self.mutation_positions,
            predictor=self.predictor,
            mutation_operator=self.mutation_operator
        )
        env = DummyVecEnv([lambda: env])

        # Initialize RL Agent
        tensorboard_log_dir = "logs/rl/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        rl_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
        rl_model.learn(total_timesteps=10000, callback=TensorboardCallback())

        # Choose the most well-mutated model
        mutated_sequence = self.mutation_operator.mutate_sequence(self.sequence)
        print(f"Mutated Sequence: {mutated_sequence}")
        analysis_results = self.analyzer.analyze_sequence(mutated_sequence, self.mutation_positions, self.expected_bases)
        for result in analysis_results:
            print(result)

        # Save models
        model_path = "models/treatment_predictor"
        rl_model_path = "models/ppo_gene_mutation"
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(rl_model_path, exist_ok=True)
        self.predictor.save_model(model_path)
        rl_model.save(rl_model_path)
        print(f"Models saved at '{model_path}' and '{rl_model_path}'")

        # Visualize Results
        self.visualize()

        # Summarize Data
        self.summarize_data()

    def visualize(self):
        plt.figure(figsize=(8,6))
        plt.scatter(np.random.rand(100), np.random.rand(100), color='green', label='Mutations')
        plt.scatter(0.5, 0.5, marker='^', color='red', label='Target')
        plt.title('Gene Mutation Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig("mutation_visualization.png")
        plt.show()
        print("Visualization saved as 'mutation_visualization.png'")

    def summarize_data(self):
        print("Data Summary:")
        print(f"Total Sequence Length: {len(self.sequence)}")
        print(f"Mutation Positions: {self.mutation_positions}")
        print(f"Expected Bases: {self.expected_bases}")
        print("TensorBoard logs available at 'logs/fit/' and 'logs/rl/'")
        print("To visualize TensorBoard, run the following commands in a new Colab cell:")
        print("```python")
        print("%load_ext tensorboard")
        print("%tensorboard --logdir logs/fit")
        print("%tensorboard --logdir logs/rl")
        print("```")

# Step 5: Run the Application

if __name__ == "__main__":
    email = "your.email@example.com"  # Replace with your email
    accession = "NC_000001"  # Example accession number; replace with a valid one from your dataset
    mutation_positions = [100, 200, 300]  # Example mutation positions
    expected_bases = {100: 'A', 200: 'G', 300: 'T'}  # Expected bases at positions
    data_path = '/content/drive/MyDrive/GeneLab/DNAsequential/human.txt'  # Update based on your Drive

    app = MainApplication(email, accession, mutation_positions, expected_bases, data_path)
    app.run()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs/fit
# %tensorboard --logdir logs/rl
