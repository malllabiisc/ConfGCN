#$ Evaluates the pre-trained ConfGCN models on Citeseer, Cora, Pubmed and Cora-ML datasets

# Evaluate on Citeseer
python conf_gcn.py -eval -restore -name best_citeseer -data citeseer

# Evaluate on Cora
python conf_gcn.py -eval -restore -name best_cora -data cora

# Evaluate on Pubmed
python conf_gcn.py -eval -restore -name best_pubmed -data pubmed

# Evaluate on Cora-ML
python conf_gcn.py -eval -restore -name best_cora_ml -data cora_ml