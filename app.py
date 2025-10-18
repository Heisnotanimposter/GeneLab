"""Streamlit web application for GeneLab."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from genelab import DNAParser, SequenceAnalyzer, GeneticMutation
from genelab.utils import generate_training_data, vectorize_features
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Page configuration
st.set_page_config(
    page_title="GeneLab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🧬 GeneLab - Bioinformatics Toolkit</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Sequence Analysis", "Mutation Simulation", "Visualization", "About"]
)

# Home Page
if page == "Home":
    st.header("Welcome to GeneLab")
    st.markdown("""
    GeneLab is a comprehensive bioinformatics toolkit for:
    - **DNA/RNA Sequence Processing**: Parse, encode, and analyze genetic sequences
    - **Genetic Mutation Simulation**: Advanced mutation operators for evolutionary algorithms
    - **Machine Learning**: Deep learning models for sequence classification
    - **Data Visualization**: Interactive visualizations of genomic data
    
    ### Quick Start
    1. Navigate to **Sequence Analysis** to analyze DNA sequences
    2. Use **Mutation Simulation** to simulate genetic mutations
    3. Explore **Visualization** for interactive data exploration
    """)
    
    st.info("💡 Tip: Upload your DNA sequences in the Sequence Analysis page to get started!")

# Sequence Analysis Page
elif page == "Sequence Analysis":
    st.header("DNA Sequence Analysis")
    
    # File upload
    st.subheader("Upload DNA Sequences")
    uploaded_file = st.file_uploader(
        "Choose a TXT file containing DNA sequences (one per line)",
        type=["txt", "fasta", "fa"]
    )
    
    # Encoding option
    col1, col2 = st.columns(2)
    with col1:
        encoding_option = st.selectbox("Choose Encoding Type", ["one-hot", "integer", "kmer"])
    with col2:
        kmer_size = st.slider("K-mer Size (for kmer encoding)", 2, 10, 6)
    
    if uploaded_file is not None:
        # Read file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # Parse sequences based on file type
        if uploaded_file.name.endswith(('.fasta', '.fa')):
            parser = DNAParser(encoding=encoding_option)
            sequences = parser.parse_fasta(uploaded_file)
            sequence_list = [seq[1] for seq in sequences]
            headers = [seq[0] for seq in sequences]
        else:
            content = stringio.read().strip().split('\n')
            sequence_list = [seq.strip().upper() for seq in content if seq.strip()]
            headers = [f"Sequence_{i+1}" for i in range(len(sequence_list))]
        
        st.success(f"✅ Loaded {len(sequence_list)} sequences")
        
        # Display sequences
        with st.expander("View Sequences"):
            df = pd.DataFrame({
                'Header': headers,
                'Sequence': sequence_list,
                'Length': [len(seq) for seq in sequence_list]
            })
            st.dataframe(df, use_container_width=True)
        
        # Analyze sequences
        st.subheader("Sequence Statistics")
        analyzer = SequenceAnalyzer()
        parser = DNAParser(encoding=encoding_option)
        
        stats_data = []
        for i, seq in enumerate(sequence_list):
            stats = parser.get_sequence_stats(seq)
            stats['Header'] = headers[i]
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Length", f"{stats_df['length'].mean():.0f}")
        with col2:
            st.metric("Average GC Content", f"{stats_df['GC_content'].mean():.2f}%")
        with col3:
            st.metric("Total Sequences", len(sequence_list))
        with col4:
            st.metric("Total Bases", stats_df['length'].sum())
        
        # Visualize statistics
        fig1 = px.bar(stats_df, x='Header', y='GC_content', 
                     title='GC Content by Sequence',
                     labels={'GC_content': 'GC Content (%)', 'Header': 'Sequence'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Encode sequences
        st.subheader("Sequence Encoding")
        if st.button("Encode Sequences"):
            with st.spinner("Encoding sequences..."):
                parser = DNAParser(encoding=encoding_option)
                encoded_sequences = parser.encode_sequences(sequence_list)
                
                st.success(f"✅ Encoded {len(encoded_sequences)} sequences")
                st.info(f"Encoded shape: {encoded_sequences.shape}")
                
                # PCA visualization
                if encoded_sequences.shape[1] >= 2:
                    st.subheader("PCA Visualization")
                    
                    # Reduce dimensions
                    n_components = min(3, encoded_sequences.shape[1])
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(encoded_sequences)
                    
                    if n_components == 3:
                        # 3D plot
                        fig = px.scatter_3d(
                            x=pca_result[:, 0],
                            y=pca_result[:, 1],
                            z=pca_result[:, 2],
                            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                            title='3D PCA of Encoded Sequences',
                            hover_name=headers
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # 2D plot
                        fig = px.scatter(
                            x=pca_result[:, 0],
                            y=pca_result[:, 1],
                            labels={'x': 'PC1', 'y': 'PC2'},
                            title='2D PCA of Encoded Sequences',
                            hover_name=headers
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Explained variance
                    explained_var = pca.explained_variance_ratio_
                    st.info(f"Explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")

# Mutation Simulation Page
elif page == "Mutation Simulation":
    st.header("Genetic Mutation Simulation")
    
    # Input sequence
    st.subheader("Input Sequence")
    input_sequence = st.text_input(
        "Enter DNA sequence",
        value="ATCGATCGATCGATCGATCGATCG",
        help="Enter a DNA sequence (A, T, C, G only)"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.01, 0.01)
    with col2:
        num_generations = st.slider("Number of Generations", 1, 100, 10)
    with col3:
        population_size = st.slider("Population Size", 10, 100, 50)
    
    if st.button("Run Mutation Simulation"):
        if input_sequence:
            with st.spinner("Simulating mutations..."):
                # Initialize mutation operator
                mutator = GeneticMutation(mutation_rate=mutation_rate)
                
                # Define fitness function (prefer high GC content)
                def fitness_function(sequence):
                    gc_count = sequence.count('G') + sequence.count('C')
                    return gc_count / len(sequence)
                
                # Create population
                population = mutator.create_population(input_sequence, population_size)
                
                # Evolve population
                final_pop, final_scores = mutator.evolve_population(
                    population,
                    fitness_function=fitness_function,
                    generations=num_generations,
                    verbose=False
                )
                
                # Get best sequence
                best_idx = np.argmax(final_scores)
                best_sequence = final_pop[best_idx]
                best_fitness = final_scores[best_idx]
                
                st.success("✅ Simulation completed!")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Sequence", input_sequence)
                    st.metric("Original GC Content", f"{fitness_function(input_sequence)*100:.2f}%")
                with col2:
                    st.metric("Best Sequence", best_sequence)
                    st.metric("Best GC Content", f"{best_fitness*100:.2f}%")
                
                # Visualize evolution
                st.subheader("Evolution Progress")
                
                # Simulate evolution progress
                progress_scores = []
                current_pop = population.copy()
                for gen in range(num_generations):
                    scores = [fitness_function(seq) for seq in current_pop]
                    progress_scores.append(max(scores))
                    
                    # Evolve one generation
                    elites = mutator.select_elites(current_pop, scores)
                    new_pop = elites.copy()
                    while len(new_pop) < len(current_pop):
                        parent1 = mutator._tournament_selection(current_pop, scores)
                        parent2 = mutator._tournament_selection(current_pop, scores)
                        child1, child2 = mutator.crossover(parent1, parent2)
                        child1 = mutator.mutate_sequence(child1)
                        child2 = mutator.mutate_sequence(child2)
                        new_pop.extend([child1, child2])
                    current_pop = new_pop[:len(population)]
                
                fig = px.line(
                    x=list(range(1, num_generations + 1)),
                    y=progress_scores,
                    labels={'x': 'Generation', 'y': 'Best Fitness (GC Content)'},
                    title='Evolution Progress'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Please enter a valid DNA sequence")

# Visualization Page
elif page == "Visualization":
    st.header("Data Visualization")
    
    st.subheader("Generate Sample Data")
    num_sequences = st.slider("Number of Sequences", 10, 100, 50)
    seq_length = st.slider("Sequence Length", 50, 500, 100)
    
    if st.button("Generate and Visualize"):
        with st.spinner("Generating data..."):
            # Generate random sequences
            bases = ['A', 'T', 'C', 'G']
            sequences = []
            for i in range(num_sequences):
                seq = ''.join(np.random.choice(bases, size=seq_length))
                sequences.append(seq)
            
            # Calculate statistics
            parser = DNAParser()
            stats_data = []
            for seq in sequences:
                stats = parser.get_sequence_stats(seq)
                stats_data.append(stats)
            
            stats_df = pd.DataFrame(stats_data)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.histogram(
                    stats_df,
                    x='GC_content',
                    nbins=20,
                    title='GC Content Distribution',
                    labels={'GC_content': 'GC Content (%)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.scatter(
                    stats_df,
                    x='A',
                    y='T',
                    size='GC_content',
                    color='GC_content',
                    title='A vs T Content',
                    labels={'A': 'A Content', 'T': 'T Content'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Composition heatmap
            st.subheader("Nucleotide Composition")
            composition = stats_df[['A', 'T', 'G', 'C']].mean()
            fig3 = px.bar(
                x=composition.index,
                y=composition.values,
                title='Average Nucleotide Composition',
                labels={'x': 'Nucleotide', 'y': 'Count'}
            )
            st.plotly_chart(fig3, use_container_width=True)

# About Page
elif page == "About":
    st.header("About GeneLab")
    st.markdown("""
    ### Version 2.0
    
    GeneLab is a comprehensive bioinformatics toolkit designed for:
    - Researchers working with genomic data
    - Students learning bioinformatics
    - Developers building bioinformatics applications
    
    ### Features
    - **Modern Python Architecture**: Clean, modular, and well-documented
    - **Machine Learning Integration**: Deep learning models for sequence analysis
    - **Interactive Web Interface**: Easy-to-use Streamlit application
    - **Comprehensive Testing**: Full test coverage with pytest
    - **Extensible Design**: Easy to add new features and modules
    
    ### Technology Stack
    - Python 3.9+
    - TensorFlow/Keras for deep learning
    - BioPython for sequence handling
    - Streamlit for web interface
    - Plotly for interactive visualizations
    
    ### License
    MIT License - Free to use and modify
    
    ### Contributing
    Contributions are welcome! Please see the README for guidelines.
    
    ### Contact
    - GitHub: https://github.com/Heisnotanimposter/GeneLab
    - Issues: https://github.com/Heisnotanimposter/GeneLab/issues
    """)
    
    st.info("💡 This is a modernized version of GeneLab with improved usability and features!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Made with ❤️ by the GeneLab Team | Version 2.0</p>",
    unsafe_allow_html=True
)
