import scanpy as sc
import pandas as pd
from simsational_api import SIMSPretrainedAPI


def test_simsational_pretraining():
    adata = sc.datasets.pbmc3k_processed()
    print(adata.obs)
    adata.obs['cell_type'] = adata.obs['louvain'].astype(str)

    print("Setting up the pretraining model")
    model = SIMSPretrainedAPI(data=adata, class_label='cell_type')

    model.setup_trainer(accelerator="cpu", devices=1)
    print("training model")
    model.train()

if __name__ == "__main__":
    test_simsational_pretraining()
