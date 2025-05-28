from collections import defaultdict
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

from .helpers import compute_tsne

class MetricsTracker:
    """
    Tracks metrics during training. Currently only makes sense to use for layers of form X(k+1)=A~X(k)W(k) and assumes that. 
    """
    def __init__(self, model, epochs, record_features=False):
        """self.epochs = epochs
        self.model = model
        self.record_features = record_features
        self.metrics = defaultdict(lambda: defaultdict(dict))  # epoch -> layer -> type -> stats TODO include representations (see below)
        self._hooks = []
        self._current_epoch = 0
        #self.modules = [module for module in self.model.named_modules() if isinstance(module, GCNConv, SGConv, SSGConv)]
        # TODO potentially one linear layer too much. is list of either ttorchgeo.nn.Linear or HeteroLinear layers in case of split weights (for 3 models listed)
        # => list of weight matrices or list of list of weight matrices (list for each layer)
        if model.shared and model.layer_type == LayerType.GCNCONV:
            self._weights = [conv.lin.weight for conv in self.model.convs]
            #self._weights.append(self.model.out_layer[1].lin.weight)
        else:
            self._weights = [hetero_lin.weights for hetero_lin in self.model.linears]

        self._convs = self.model.convs
        #self._convs.append(self.model.out_layer[1])

        # currently (in particular kmeans & random projections) this will be the same for all layers 
        self._activations = self.model.act_layers

        self.writer = SummaryWriter(f"ablation_experiments/Layers: {self.model.num_layers} Clusters: {self.model.num_clusters} Clustering: {self.model.cluster_config['strat']} Activations shared: {str(self.model.shared)} Tokenization: {str(self.model.tokenize)} hard_assignent: {self.model.cluster_config['hard_assignment']}" )

        # self._register_metric_hooks()"""
        pass
       




    def _compute_metrics(self, matrix, weight=False):
        pass


    def _register_gradient_hooks(self):
        # just use attributes

        for layer, conv in enumerate(self._convs): 
            hook = conv.register_forward_hook(self._metric_hook(layer))
            self._hooks.append(hook)
       
       
    # TODO only execute or register this for last epoch or in eval mode at the end
    def _register_embedding_hooks(self):
        for layer, activation in enumerate(self._activations): 
            hook = activation.register_forward_hook(self._embedding_hook(layer))
            self._hooks.append(hook)


    def _metric_hook(self, layer):
        def hook(module, inputs, output):
            if module.train: 
                if isinstance(output, tuple):  # apparently  some PyG layers
                    output = output[0]
                    # defaultdict?
                self.metrics[self._current_epoch]['metrics'][layer] = self._compute_metrics(output.detach().cpu())
        return hook
    


    def _embedding_hook(self, layer): 
        def hook(module, inputs, output):
            if module.train:
                x = output[0].detach().cpu()
                self.metrics[self._current_epoch]['embeddings'][layer] = compute_tsne(x)
        return hook



    def _record_weights(self):
    # just use attributes => only for GNN pr self.shared != True!
        
        for layer, w in enumerate(self._weights): 
            if self.model.shared : 
                self.metrics[self._current_epoch]['weights'][layer] = self._compute_metrics(w.detach().cpu(), weight=True)

            else: 

                self.metrics[self._current_epoch]['weights'][layer] = {}
                for cluster_id in range(self.model.num_clusters):
                    self.metrics[self._current_epoch]['weights'][layer][cluster_id] = self._compute_metrics(w[cluster_id].detach().cpu(), weight=True)





    def log_to_board(self, loss, acc): 
        
        pass
        



        




    