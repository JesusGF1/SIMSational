import scanpy as sc
import pandas as pd
from simsational_api import SIMSPretrainedAPI
from pytorch_lightning.loggers import WandbLogger
import umap
import numpy as np
import matplotlib.pyplot as plt

def test_simsational_pretraining():
    adata = sc.datasets.pbmc3k_processed()
    #write adata to h5ad
    #adata.write("pbmc3k_processed.h5ad")

    print(adata.obs)
    print(adata.var)
    adata.obs['cell_type'] = adata.obs['louvain'].astype(str)
    print("Setting up the pretraining model")
    model = SIMSPretrainedAPI(data=adata, class_label='cell_type', batch_size=8)
    logger = WandbLogger(name="simsational_pretraining",project="simsational")
    model.setup_trainer(accelerator="cpu", devices=1,limit_train_batches=1,limit_val_batches=1,max_epochs=10,
                        logger=logger)
    print("training model")
    model.train()
    print("model trained")
    #Now model predict to get embeddings
    
    embeddings = model.get_embeddings(adata)
    print("Embeddings")
    print(embeddings)
    print(embeddings.shape)
    print("UMAP")
    embeddings = embeddings.detach().numpy()
    umap_embeddings = umap.UMAP().fit_transform(embeddings)
    print(umap_embeddings)
    print(umap_embeddings.shape)
    #Now plot the UMAP
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    plt.figure(figsize=(10,10))
    plt.scatter(umap_embeddings[:,0], umap_embeddings[:,1], c=adata.obs['louvain'].cat.codes, cmap='tab20')
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig("umap_embeddings.png")
if __name__ == "__main__":
    test_simsational_pretraining()
