from collections import defaultdict
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

from .metrics import to_dense_adj, compute_mad, compute_GMAD, compute_dirichlet_energy, compute_mad_gap
from .utils import compute_tsne, prop_over_epoch, discretized_changes
from .plot_data import plot_with_centers
from .model_params import LayerType


class MetricsTracker:
    """
    Tracks metrics during training. Currently only makes sense to use for layers of form X(k+1)=A~X(k)W(k) and assumes that. 
    """
    def __init__(self, model, epochs, record_features=False):
        self.epochs = epochs
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

        # self._register_metric_hooks()
       




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


    def _record_assignments(self): 
        """Records cluster assignments of features over epochs to e.g gauge clustering dynamics/stability."""
        # record for DifferentiableClustering also hard_assignments
      
        hard_assignments = []
        cluster_indices = []
        for _, activation in enumerate(self._activations): 
            d = activation.get_assignments()
            hard_assignments.append(d["hard_assignments"])
            cluster_indices.append(d["hard_assignments"])
            

        self.metrics[self._current_epoch]["hard_assignments"] = hard_assignments
        self.metrics[self._current_epoch]["cluster_indices"] = cluster_indices


    def log_to_board(self, loss, acc): 
        # log to tensor board, call every epoch (self.writer)
        # dont need plot for this (for checking training behavior for now only, later for sure want to store)
        self.writer.add_scalar("Training loss", loss, self._current_epoch)
        self.writer.add_scalar("Training accuray", acc, self._current_epoch)
        # for logging, last epoch all layers, here current epoch last layer
        """last_layer = self.model.num_layers-1
        self.writer.add_scalar("Dirichlet energy", self.metrics[self._current_epoch]["metrics"][last_layer]["dirichlet_energy"], self._current_epoch)
        self.writer.add_scalar("MAD", self.metrics[self._current_epoch]["metrics"][last_layer]["mad"], self._current_epoch)
        self.writer.add_scalar("rank", self.metrics[self._current_epoch]["metrics"][last_layer]["rank"], self._current_epoch)
        self.writer.add_scalar("sv", self.metrics[self._current_epoch]["metrics"][last_layer]["sv"], self._current_epoch)"""

        
        



        




    def _log(self): 
        # write to logfile the accuracies/losses, other metrics like rank etc. 
        # plot later (for comparison between models/architectures/experiments)
        # TODO: we want metrics only for last layer during trainig to see evolution over epochs + for all layers in final
        # epoch for plotting against other experiments -> modify + TODO add clustering stability metrics
        
        # log to file for post processing => execute_plotting script
        # log rank, sv, etc.. of last epoch over all layers for feature matrix and weights to tensorboard
        



        ep = self._current_epoch - 1
        for layer in range(len(self._weights)): 
            if not self.model.shared: 
                for cluster in range(self.model.num_clusters): 
                    rank_wr = self.metrics[ep]["weights"][layer][cluster]["rank"]
                    self.writer.add_scalar(f"weight matrix rank over layer, cluster {cluster}", rank_wr, layer)
            else: 
                rank_wr = self.metrics[ep]["weights"][layer]["rank"]
                self.writer.add_scalar(f"weight matrix rank over layer, cluster", rank_wr, layer)




        for layer in range(self.model.num_layers): 
            rank_fr = self.metrics[ep]["metrics"][layer]["rank"]
            self.writer.add_scalar(f"feature matrix rank over layer", rank_fr, layer) 




        # log assignment metric to evaluate clustering stability (also soft assignments?) over epochs!
        for step, swaps in enumerate(discretized_changes(prop_over_epoch(self.metrics, attribute="hard_assignments"), len(self._activations))): 
            for layer, swap in enumerate(swaps): 
                # need to multiply with "stepsize"
                self.writer.add_scalar(f"average cluster assignment changes for layer {layer}", swap, step)

        # plot clusters for last layer TODO: later separate this into ModelTrainer and use Visualizer class for plotting 



        if ep == self.epochs-1: 
            assignments = self.metrics[ep]["hard_assignments"]
            representations = self.metrics[ep]["embeddings"]

            for layer in range(len(self._activations)): 

                fig = plot_with_centers(model=self.model, representations=representations[layer], hard_assignments=assignments[layer], layer=layer)
                self.writer.add_figure(f"Representations Epoch: {ep} Layer: {layer}", fig)
            






    def step_epoch(self):
        # self._record_weights()
        self._record_assignments()
        self._current_epoch += 1
        if (self._current_epoch == self.epochs-1): #and self.record_features:
            # collect embeddings for last layer
            self._register_embedding_hooks()
            self._record_weights()
            self._register_metric_hooks()


    def get_all_metrics(self):
        return dict(self.metrics)

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._log()

    

    # call record_weights, record_assignments, (potentially compute_embeddings), (_log_to_board, step_epoch
    # at end of training call get_all_metrics, log, and cleanup