# Step 1: Import necessary libraries
!pip install biopython
import torch
import torch.nn as nn
import torch.optim as optim

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Step 2: Create a sample data file with real protein sequences
sample_data = '''>Human_Insulin
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTR
>Human_p53_Tumor_Suppressor
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTED
>Human_Hemoglobin
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK
>Human_Myoglobin
MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE
>Human_Lysozyme_C
MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRAT
'''

# Write the sample data to a file
with open("protein_sequences.fasta", "w") as file:
    file.write(sample_data)

# Step 3: Load and display sequences using BioPython
sequences = list(SeqIO.parse("protein_sequences.fasta", "fasta"))
print("Loaded protein sequences:")
for seq_record in sequences:
    print(seq_record.id, seq_record.seq[:50])  # display first 50 amino acids for brevity

# Step 4: Define a simple neural network for sequence classification (assuming further processing)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(50, 100)  # Example dimension adjustment
        self.fc2 = nn.Linear(100, 2)   # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNN()

# Step 5: Encode the protein sequences numerically for input into the neural network
# This could involve converting amino acids to integers or using a more complex encoding
encoded_sequences = []
for seq_record in sequences:
    encoded_seq = [ord(c) % 20 for c in seq_record.seq]  # Simplified encoding for demonstration
    encoded_sequences.append(encoded_seq[:50])  # Only first 50 amino acids for uniform input size

encoded_tensor = torch.tensor(encoded_sequences, dtype=torch.float32)

# Step 6: Perform a forward pass with the model
output = model(encoded_tensor)
print("Output from the neural network:")
print(output)
