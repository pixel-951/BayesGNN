from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops, to_dense_adj, to_undirected
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torchmetrics import CalibrationError 

import wandb


from ..utils.datasets import load_dataset
from ..utils.metrics import brier_score, brier_over_under, expected_calibration_error
from ..utils.metrics_tracker import MetricsTracker

from ..model.network import Net
from .ivon import IVON



class ModelTrainer:

    def __init__(self, config):

        self._raw_config = config

        wandb.init(
            project="optimizer-ablation",
            dir="../../wandb_logs",
            config=config
        )

        print(f"config: {config}")

        self.config = wandb.config

        self.epochs = self.config["epochs"]
        self.lr = self.config["lr_model"]
        self.weight_decay = self.config["weight_decay"]
        self.optimizer_type = self.config["optimizer_type"]
        self.lr_scheduling = False
        if "lr_scheduler" in self.config: 
            self.lr_scheduling = True
        
        #self.experiment_number = self.config["experiment_number"]

        if self.optimizer_type == "ivon":
            self.train_samples = self.config.get("train_samples", 1)
            self.test_samples = self.config.get("test_samples", 20)
            self.ess = self.config.get("ess", 1.0)
            self.hess_init = self.config.get("hess_init", 1.0)
            self.hess_approx = self.config.get("hess_approx", "price")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_name = self.config["dataset_name"]
        self.dataset_type = self.config["dataset_type"]

        self.hidden_features = self.config["hidden_features"]
        self.layer_type = self.config["layer_type"]
        self.num_layers = self.config["num_layers"]

        self.batch_train = False
        self.max_val_acc = 0

        

        print(config)

        self.load_dataset()
        self.test_ece = CalibrationError(task="multiclass", n_bins=10, num_classes=self.num_classes)

        self.setup_model()



    def load_dataset(self) -> None:

        dataset = load_dataset(
            self.dataset_name,
            dataset_type=self.dataset_type,
        )
        """
        self.dataset, self.num_classes, self.input_features = (
            dataset,
            self.dataset.num_classes,
            (
                self.dataset.num_features
                if self.dataset_type == "Plantoid"
                else dataset[0]
            ),
            dataset[1],
            torch.Tensor(self.dataset.x).shape[1],
        )
        self.data = self.dataset[0].to(self.device)
        self.edge_index = (
            to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)
            if self.dataset_type == "PygNodePropPredDataset"
            else self.data.edge_index
        )"""
        """if self.dataset_type == "Planetoid":"""
        self.dataset = dataset
        self.num_classes = self.dataset.num_classes
        self.input_features = self.dataset.num_features
        """else:
            self.dataset = dataset[0]
            self.num_classes = dataset[1]
            self.input_features = torch.Tensor(self.dataset.x).shape[1]"""

        self.data = self.dataset[0].to(self.device)
        self.num_samples = self.data.x.shape[0]
        self.edge_index = (
            to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)
            if self.dataset_type == "PygNodePropPredDataset"
            else self.data.edge_index
        )

    def setup_model(self) -> None:

        self.model = Net(
            input_features=self.input_features,
            output_features=self.num_classes,
            hidden_features=self.hidden_features,
            num_layers=self.num_layers,
            layer_type=self.layer_type,
        ).to(self.device)

        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                [{"params": self.model.parameters()}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            # TODO add lr sheduling
            self.optimizer = IVON(
                self.model.parameters(), lr=self.lr, ess=self.num_samples*self.ess, hess_init=self.hess_init, hess_approx=self.hess_approx
            )
            if self.lr_scheduling: 
                #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)
                self.scheduler = self.get_ivon_scheduler(self.optimizer, self.epochs)

    def train(self) -> None:

        print(self.model)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The number of required params : {num_params}")

        """if self.dataset_type == "PygNodePropPredDataset":
            evaluator = Evaluator(self.dataset_name)
            x = self.data.x.to(self.device)
            y_true = self.data.y.to(self.device)
            edge_index = self.data.edge_index.to(self.device)
            edge_index = to_undirected(edge_index, num_nodes=self.data.num_nodes)
        else:
            edge_index = self.data.edge_index"""

        for epoch in tqdm(range(self.epochs)):
            """def closure():
                with self.optimizer.sampled_params(train=True):
                    self.optimizer.zero_grad()
                    _, out = self.model(data.x, edge_index)
                    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    return loss
            self.optimizer.step(closure)"""

            """if self.batch_train:
                train_loader = NeighborLoader(
                    data,
                    input_nodes=data.train_mask,
                    num_neighbors=[25, 10],
                    shuffle=True,
                )
                train_acc, train_loss = self._train_batches(train_loader, epoch)
                val_acc, val_loss = self._valid(data)
                test_acc, test_loss = self._test(data)"""

            train_loss = self._train(self.data, self.edge_index)
            train_acc, val_acc, test_acc, ece, over_ece, under_ece, brier, entropy, nll = self._evaluate(self.data, self.edge_index)
            
            # TODO implement early stopping here
            self.max_val_acc = max(self.max_val_acc, val_acc)

            log = {
                # Accuracy metrics
                "train/accuracy": train_acc,
                "val/accuracy": val_acc,
                "test/accuracy": test_acc,
                
                # Calibration metrics
                "test/ece": ece,
                "test/ece_over": over_ece,
                "test/ece_under": under_ece,
                "test/brier_score": brier,
                "test/entropy": entropy,
                "test/nll": nll,
                
                # Training metrics
                "train/loss": train_loss,
                "max_val_acc": self.max_val_acc
                
            }
            if self.lr_scheduling: 
                self.scheduler.step()
                log.update({
                    #"mean": self.optimizer.param_avg, 
                    #"variance": self.optimizer.noise
                    "lr_sched": self.scheduler.get_last_lr()[0]
                })
                print(self.scheduler.get_last_lr()[0], self.scheduler.get_lr()[0])
            wandb.log(log, step=epoch)

        wandb.finish()

    def _train(self, data, edge_index):
        self.model.train()
        losses = []

        if self.optimizer_type == "ivon":
            for _ in range(self.train_samples):
                self.optimizer.zero_grad()
                with self.optimizer.sampled_params(train=True):
                    logit = self.model(data.x, edge_index)
                    log_probs = F.log_softmax(logit, dim=1)
                    loss = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            return sum(losses) / len(losses)

        elif self.optimizer_type == "adam":
            self.optimizer.zero_grad()
            logit = self.model(data.x, edge_index)
            log_probs = F.log_softmax(logit, dim=1)
            loss = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            return loss.item()

    @torch.no_grad()
    def _evaluate(self, data, edge_index):
        self.model.eval()

        if self.optimizer_type == "ivon":
            sampled_probs = []
            for _ in range(self.test_samples):
                with self.optimizer.sampled_params():
                    sampled_logit = self.model(data.x, edge_index)
                    sampled_probs.append(F.softmax(sampled_logit, dim=1))
            sm = torch.mean(torch.stack(sampled_probs), dim=0)
            pred = sm.argmax(dim=1)
            log_probs = torch.log(sm + 1e-12)

        elif self.optimizer_type == "adam":
            logit = self.model(data.x, edge_index)
            sm = F.softmax(logit, dim=1)
            pred = sm.argmax(dim=1)
            log_probs = F.log_softmax(logit, dim=1)

        #ece = self.test_ece(sm[data.test_mask], data.y[data.test_mask])
        brier = brier_score(sm[data.test_mask], data.y[data.test_mask])
        #b_over, b_under = brier_over_under(sm[data.test_mask], data.y[data.test_mask])
        ece, upper_ece, under_ece = expected_calibration_error(sm[data.test_mask], data.y[data.test_mask])
        entropy = -(sm * torch.log(sm + 1e-12)).sum(dim=1)
        mean_entropy = entropy[data.test_mask].mean().item()

        nll = F.nll_loss(log_probs[data.test_mask], data.y[data.test_mask]).item()

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        return train_acc.item(), val_acc.item(), test_acc.item(), ece, upper_ece, under_ece, brier, mean_entropy, nll



    



    def _train_batches(self, train_loader, current_epoch):
        """
        Perform batch training for one epoch.

        Parameters:
            train_loader (torch_geometric.data.NeighborLoader): DataLoader for training.

        Returns:
            tuple: Tuple containing training accuracy and loss.
        """
        self.model.train()
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f"Epoch {current_epoch:02d}")
        total_loss = total_correct = total_examples = 0

        for batch in train_loader:
            if batch.num_nodes >= self.clusters:
                self.optimizer.zero_grad()
                y = batch.y[: batch.batch_size]
                _, y_hat = self.model(batch.x, batch.edge_index.to(self.device))[
                    : batch.batch_size
                ]
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * batch.batch_size
                total_correct += int((y_hat.argmax(dim=-1) == y).sum())
                total_examples += batch.batch_size
                pbar.update(batch.batch_size)

        pbar.close()
        train_acc, train_loss = (
            total_loss / total_examples,
            total_correct / total_examples,
        )

        return train_acc, train_loss





    def get_ivon_scheduler(self, optimizer, total_epochs, warmup_epochs=400):
        
        linear = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs) #-warm_epochs
        
        #scheduler = SequentialLR(optimizer, [linear, cosine])
        return cosine













    @torch.no_grad()
    def test_ogb(self, edge_index, data, evaluator):
        self.model.eval()
        _, out = self.model(data.x, edge_index)
        y_pred = out.argmax(dim=-1, keepdim=True)

        y_true = data.y.unsqueeze(1)
        train_acc = evaluator.eval(
            {
                "y_true": y_true[data.train_mask],
                "y_pred": y_pred[data.train_mask],
            }
        )["acc"]
        valid_acc = evaluator.eval(
            {
                "y_true": y_true[data.val_mask],
                "y_pred": y_pred[data.val_mask],
            }
        )["acc"]
        test_acc = evaluator.eval(
            {
                "y_true": y_true[data.test_mask],
                "y_pred": y_pred[data.test_mask],
            }
        )["acc"]

        return train_acc, valid_acc, test_acc
