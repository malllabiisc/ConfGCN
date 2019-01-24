echo "Create log directory"
mkdir log

echo "Extracts Citeseer, Cora, Pubmed, and Cora-ML datasets"
unzip data.zip

echo "Extracts best saved models on all four datasets"
unzip checkpoints.zip

echo "Install python dependencies"
pip install -r requirements.txt