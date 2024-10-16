from pytorch_tabnet.tab_network import TabNetPretraining
from pytorch_tabnet.utils import create_group_matrix, create_explain_matrix
from pytorch_tabnet.metrics import UnsupervisedLoss
import pytorch_lightning as pl
import torch
from typing import Callable, List, Union
from dataclasses import field
import anndata as an
from scipy.sparse import csr_matrix
import os
from simsational.data import CollateLoader
from simsational.inference import DatasetForInference
import numpy as np
from tqdm import tqdm


class SIMSPretraining(pl.LightningModule):
    def __init__(self,
        #All of these come straight from the pretrainer class
        input_dim,
        pretraining_ratio=0.2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", #Maybe entmax? 
        n_shared_decoder=1,
        n_indep_decoder=1,
        
        #Things inherited from SIMS
        genes: list[str] = None,
        #cells: list[str] = None,
        label_encoder: Callable = None,

        #You can group features to consider them as a simmilar feature. We could use this for gene modules
        #But for our first implementation we will not use this. Default to None for now
        grouped_features: List[List[int]] = [],

        loss: Callable = None, #Defaults to UnsupervisedLoss
        optim_params=None, #Defaults to AdamW 
        scheduler_params=None, #Defaults to ReduceLROnPlateau
        ):
        super(SIMSPretraining, self).__init__()
        self.save_hyperparameters()
        self.genes = genes
        self.label_encoder = label_encoder

        # Stuff needed for training
        self.input_dim = input_dim

        #Add loss
        self.loss = loss
        #Defaults to UnsupervisedLoss
        if loss is None:
            self.loss = UnsupervisedLoss
        
        #Add metrics/ Log embeddings and then have a function to calculate the ARI
        self.metrics = {
            "train": {},
            "val": {},
        }
        #Add optim params, have to check if the same ones work for the pretraining network
        self.optim_params = (
            optim_params
            if optim_params is not None
            else {
                "optimizer": torch.optim.AdamW,
                "lr": 0.001,
                "weight_decay": 0.01,
            }
        )
        #Add scheduler params
        self.scheduler_params = (
            scheduler_params
            if scheduler_params is not None
            else {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "factor": 0.75,  # Reduce LR by 25% on plateau
            }
        )

        #This creates group matrices for the pretraining network
        self.grouped_features = grouped_features #This can not be none
        print(f"Grouped features: {self.grouped_features}")
        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)
    
        
        #Initialize network

        # Define the network and add all the inputs that you need from the Init
        self.network = TabNetPretraining(
            input_dim=input_dim,
            pretraining_ratio=pretraining_ratio,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            n_shared_decoder=n_shared_decoder,
            n_indep_decoder=n_indep_decoder,
            group_attention_matrix=self.group_matrix,
        )

        #Initialize explainabililty matrix
        self.reducing_matrix = create_explain_matrix(
        self.network.input_dim,
        self.network.cat_emb_dim,
        self.network.cat_idxs,
        self.network.post_embed_dim,
        )
        #Set device


        return
    
    def forward(self, x):
        """
        res : output of reconstruction
        embedded_x : embedded input
        obf_vars : which variable where obfuscated, in our case which genes were obfuscated
        """
        res, embedded_x, obf_vars = self.network(x)
        return res, embedded_x, obf_vars
    

    def _step(self, batch,tag):
        x,y = batch #X is the only thing we need for pretraining
        res, embedded_x, obf_vars = self.forward(x)
        loss = self.loss(res, x, obf_vars)

        #Use downstream metrics like ARI to validate the model clustering capacity.
        return {
            "loss": loss
        }
    
    def training_step(self, batch, batch_idx):
        results = self._step(batch, "train")
        self.log(f"train_loss", results["loss"], on_epoch=True, on_step=True)
        #Log embeddings in a list and keep...

        # for name, metric in self.metrics["train"].items():
        #     self.log(f"train_{name}", metric, on_epoch=True, on_step=False)
        return results["loss"]


    def on_train_epoch_end(self) -> None:
        #for data in dataloader
        #Get embeddings
        #Train small KNN

        # for name,metric in self.metrics["train"].items():
        #     self.log(f"train_{name}", metric, on_epoch=True, on_step=False)
        return
     
    def validation_step(self, batch, batch_idx):
        results = self._step(batch, "val")
        self.log(f"val_loss", results["loss"], on_epoch=True, on_step=True)
        # for name, metric in self.metrics["val"].items():
        #     value = metric(results["probs"], batch[1])
        #     self.log(f"val_{name}", value=value)
        return results["loss"]
    
    def on_validation_epoch_end(self):
        return
    

    def configure_optimizers(self):
        if "optimizer" in self.optim_params:
            optimizer = self.optim_params.pop("optimizer")
            optimizer = optimizer(self.parameters(), **self.optim_params)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), **self.optim_params)
        print(f"Initializing with {optimizer = }")

        if self.scheduler_params is not None:
            scheduler = self.scheduler_params.pop("scheduler")
            scheduler = scheduler(optimizer, **self.scheduler_params)
            print(f"Initializating with {scheduler = }")

        if self.scheduler_params is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def _parse_data(
            self,
            inference_data,
            batch_size=64,
            num_workers=os.cpu_count(),
            rows=None,
            **kwargs
        ) -> torch.utils.data.DataLoader:

        if isinstance(inference_data, str):
            inference_data = an.read_h5ad(inference_data)
        
        # handle zero inflation or deletion
        #TODO: Increase speed and memory consumption of zero inflation
        inference_genes = list(inference_data.var_names)
        training_genes = list(self.genes)

        # more inference genes than training genes
        assert len(set(inference_genes).intersection(set(training_genes))) > 0, "inference data shares zero genes with training data, double check the string formats and gene names"

        left_genes = list(set(inference_genes) - set(training_genes))  # genes in inference that aren't in training
        right_genes = list(set(training_genes) - set(inference_genes))  # genes in training that aren't in inference 
        intersection_genes = list(set(inference_genes).intersection(set(training_genes))) # genes in both

        if len(left_genes) > 0:
            inference_data = inference_data[:, intersection_genes].copy()
        if len(right_genes) > 0:
            print(f"Inference data has {len(right_genes)} less genes than training; performing zero inflation.")

            #zero_inflation = an.AnnData(X=np.zeros((inference_data.shape[0], len(right_genes))), obs=inference_data.obs)
            zero_inflation = an.AnnData(X=csr_matrix((inference_data.shape[0], len(right_genes))),obs=inference_data.obs)
            zero_inflation.var_names = right_genes
            inference_data = an.concat([zero_inflation, inference_data], axis=1)

        # now make sure the columns are the correct order
        inference_data = inference_data[:, training_genes].copy()

        if isinstance(inference_data, an.AnnData):
            inference_data = DatasetForInference(inference_data.X[rows, :] if rows is not None else inference_data.X)

        if not isinstance(inference_data, torch.utils.data.DataLoader):
            inference_data = CollateLoader(
                dataset=inference_data,
                batch_size=batch_size,
                num_workers=num_workers,
                **kwargs,
            )

        return inference_data
    
    def explain(self, x):
        return
    
    def predict(self, x):
        return
    
    # def get_embeddings(self, x):
    #     res, embedded_x, obf_vars = self.network(x)
    #     return embedded_x
    
    def embedding_step(self, batch, batch_idx):
        x = batch
        embedded_x = self.network.embedder(x)
        steps_out, _ = self.network.encoder(embedded_x)
        #res, embedded_x, obf_vars = self.network(x)
        #res, embedded_x, obf_vars = self.forward(x)
        return steps_out

    def get_embeddings(self, inference_data: Union[str, an.AnnData, np.array], batch_size=32, num_workers=4, rows=None, currgenes=None, refgenes=None, **kwargs):
        print("Parsing inference data...")
        loader = self._parse_data(
            inference_data,
            batch_size=batch_size,
            num_workers=num_workers,
            rows=rows,
            currgenes=currgenes,
            refgenes=refgenes,
            **kwargs
        )

        prev_network_state = self.network.training
        self.network.eval()

        embeddings = []
        batch_size = loader.batch_size
        for idx, X in enumerate(tqdm(loader)):
            embeddings.append(self.embedding_step(X, idx))
        breakpoint()
        embeddings = torch.cat(embeddings, dim=0)

        # if network was in training mode before inference, set it back to that
        if prev_network_state:
            self.network.train()

        return embeddings




    #Save the pretrained model
    def save_model(self):
        """Save the model as a ckpt file"""
        return

