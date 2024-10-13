# Step 1: Import necessary libraries (Step 3)
#!pip install biopython
import torch
import torch.nn as nn
import torch.optim as optim

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Step 2: Create a sample data file with DNA sequences
sample_data = '''>Sequence1
ATCGATCGATCGATCGATCGATCGATCGATCG
>Sequence2
CGATCGATCGATCGATCGATCGATCGATCGAT
'''

# Write the sample data to a file (simulating a file in Colab)
with open("sample_sequences.fasta", "w") as file:
    file.write(sample_data)

# Step 3: Load and display sequences using BioPython
sequences = list(SeqIO.parse("sample_sequences.fasta", "fasta"))

# Display the sequences
print("Loaded sequences:")
for seq_record in sequences:
    print(seq_record.id, seq_record.seq)

# Step 4: Define a simple neural network for sequence classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32, 64)  # assuming fixed length of input sequences
        self.fc2 = nn.Linear(64, 2)   # binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNN()

# Step 5: Encode the DNA sequences numerically for input into the neural network
# Here we simply convert A, T, C, G to 0, 1, 2, 3 as a placeholder
# Normally, you'd use a more sophisticated encoding like one-hot or k-mer based encoding
encoded_sequences = []
for seq_record in sequences:
    encoded_seq = [0 if nucleotide == 'A' else 1 if nucleotide == 'T' else 2 if nucleotide == 'C' else 3 for nucleotide in seq_record.seq]
    encoded_sequences.append(encoded_seq)

encoded_tensor = torch.tensor(encoded_sequences, dtype=torch.float32)

# Step 6: Perform a forward pass with the model (this example doesn't involve training)
output = model(encoded_tensor)
print("Output from the neural network:")
print(output)
