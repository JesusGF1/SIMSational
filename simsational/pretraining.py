from pytorch_tabnet.tab_network import TabNetPretraining
from pytorch_tabnet.utils import create_group_matrix, create_explain_matrix
from pytorch_tabnet.metrics import UnsupervisedLoss
import pytorch_lightning as pl
import torch
from typing import Callable


class SIMSPretraining(pl.LightningModule):
    def __init__(self,
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
        mask_type="sparsemax",
        n_shared_decoder=1,
        n_indep_decoder=1,
        group_attention_matrix=None,

        loss: Callable = None, #Defaults to UnsupervisedLoss
        optim_params=None, #Defaults to Adam 
        scheduler_params=None, #Defaults to ReduceLROnPlateau
        ):
        super(SIMSPretraining, self).__init__() 
        
        #Add loss
        self.loss = loss
        #Defaults to UnsupervisedLoss
        if loss is None:
            self.loss = UnsupervisedLoss
        
        #Add metrics
        self.metrics = {
            "train": {},
            "val": {},
        }
        #Add optim params, have to check if the same ones work for the pretraining network
        self.optim_params = (
            optim_params
            if optim_params is not None
            else {
                "optimizer": torch.optim.Adam,
                "lr": 0.01,
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
        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)
        #Initialize network

        # Define the network and add all the inputs that you need
        self.network = TabNetPretraining(
    
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
    

    def _step(self, batch):
        return
    
    def training_step(self, batch, batch_idx):
        return


    def on_train_epoch_end(self):
        return
    
    def validation_step(self, batch, batch_idx):
        return
    
    def on_validation_epoch_end(self):
        return
    

    def configure_optimizers(self):
        return
    
    def _parse_data():
        return
    
    def explain(self, x):
        return
    
    def predict(self, x):
        return
    
    #Save the pretrained model
    def save_model(self):
        """Save the model as a ckpt file"""
        return

