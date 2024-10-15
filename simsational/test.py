import scanpy as sc
import pandas as pd
from simsational_api import SIMSPretrainedAPI
from pytorch_lightning.loggers import WandbLogger

def test_simsational_pretraining():
    adata = sc.datasets.pbmc3k_processed()
    print(adata.obs)
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
if __name__ == "__main__":
    test_simsational_pretraining()
