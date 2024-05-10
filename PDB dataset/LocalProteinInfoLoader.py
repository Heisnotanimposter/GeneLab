import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LocalProteinInfoLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.dataframe = None

    def load_info(self, encoding='utf-8'):
        """Attempts to load the CSV file with the specified encoding, defaults to utf-8, retries with latin1 on failure."""
        try:
            self.dataframe = pd.read_csv(self.csv_file_path, encoding=encoding)
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}. Trying with 'latin1' encoding.")
            self.dataframe = pd.read_csv(self.csv_file_path, encoding='latin1')
        except Exception as e:
            print(f"Error loading data: {e}")

    def display_info(self):
        """Displays the DataFrame if it has been loaded, else prompts to load data."""
        if self.dataframe is not None:
            print(self.dataframe)
        else:
            print("Data not loaded. Please call load_info() first.")

    def filter_data(self, column, condition):
        """Filters the DataFrame based on a condition applied to the specified column."""
        if self.dataframe is not None:
            return self.dataframe.query(f"{column} {condition}")
        else:
            print("Data not loaded. Please load data before filtering.")
            return None

    def show_basic_stats(self):
        """Displays basic statistics for the DataFrame."""
        if self.dataframe is not None:
            print(self.dataframe.describe())
        else:
            print("Data not loaded. Please call load_info() first.")

    def plot_data(self, column):
        """Generates a histogram of the specified column in the DataFrame."""
        if self.dataframe is not None and column in self.dataframe.columns:
            sns.histplot(self.dataframe[column], kde=True)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f"Column '{column}' not found or data not loaded.")

# Example Usage
csv_file_path = '/path/to/yourfile.csv'  # Update this to your actual file path
loader = LocalProteinInfoLoader(csv_file_path)
loader.load_info()
loader.display_info()
filtered_data = loader.filter_data('ProteinID', '== "P01234"')
print(filtered_data)
loader.show_basic_stats()
loader.plot_data('ProteinConcentration')