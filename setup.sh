# Install python dependencies
pip install -r requirements.txt

# Extracts Citeseer, Cora, Pubmed, and Cora-ML datasets
unzip data.zip

# Extracts best saved models on all four datasets
unzip checkpoints.zip

# Create log directory
mkdir log