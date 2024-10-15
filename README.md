# ASGCL: Adaptive Sparse Mapping-based Graph Contrastive Learning Network for Cancer Drug Response Prediction

ASGCL code.

## Requirements
python==3.8
torch1.13.1，
torch-geometric2.5.3

## GDSC

- model.py: Code that implements the model.
- utils.py: The code that implements the tool.
- sampler.py: The code that implements the sampler.
- The directory processed_data contains the data required for the experiment.
  - cell_drg.csv records the log IC50 association matrix of cell line-drug. 
  - cell_drugbinary.csv records the binary cell line-drug association matrix. 
  - cellcna.csv records the CNA features of the cell line. 
  - cell_gene.csv records cell line gene expression features. 
  - cell_mutation.csv records somatic mutation features of cell lines. 
  - drug_feature.csv records the fingerprint features of drugs. 
  - null_mask.csv records the null values in the cell line-drug association matrix. 
  - threshold.csv records the drug sensitivity threshold.

└─CCLE
    │  main.py
    │  
    ├─api
    │      model.py
    │      plot.py
    │      sampler.py
    │      utils.py
    │      
    ├─processed_data
    │      cell_drug_binary.csv
    │      cell_gene.csv
    │      drug_feature.csv
    │      null_mask.csv
    │      
    └─result_data
            predict_data.csv
            true_data.csv
            


## CCLE

Same as directory GDSC.

- CCLE/processed_data/ 
  - cell_drug.csv records the log IC50 association matrix of cell line-drug. 
  - cell_drug_binary.csv records the binary cell line-drug association matrix. 
  - cell_cna.csv records the CNA features of the cell line. 
  - drug_feature.csv records the fingerprint features of drugs. 
  - cell_gene.csv records cell line gene expression features. 
  - cell_mutation.csv records somatic mutation features of cell lines.

└─GDSC
    │  main.py
    │  
    ├─api
    │      model.py
    │      plot.py
    │      sampler.py
    │      utils.py
    │      
    ├─processed_data
    │      cell_cna.csv
    │      cell_gene.csv
    │      cell_drug.csv
    │      cell_drug_binary.csv
    │      cell_mutation.csv
    │      drug_feature.csv
    │      
    └─result_data
            predict_data.csv
            true_data.csv






