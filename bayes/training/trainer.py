from tqdm import tqdm
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops, to_dense_adj, to_undirected
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import CalibrationError 


from ..utils.datasets import load_dataset
from ..utils.metrics import brier_score
from ..utils.metrics_tracker import MetricsTracker

from ..model.network import Net
from .ivon import IVON



class ModelTrainer:

    def __init__(self, config):
        self.config = config

        self.epochs = config["epochs"]
        self.lr = config["lr_model"]
        self.weight_decay = config["weight_decay"]
        self.optimizer_type = config["optimizer_type"]
        self.experiment_number = config["experiment_number"]

        if self.optimizer_type == "ivon":
            self.train_samples = config["train_samples"]
            self.test_samples = config["test_samples"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_name = config["dataset_name"]
        self.dataset_type = config["dataset_type"]

        self.hidden_features = config["hidden_features"]
        self.layer_type = config["layer_type"]
        self.num_layers = config["num_layers"]

        self.batch_train = False

        

        print(config)

        self.load_dataset()
        self.test_ece = CalibrationError(task="multiclass", n_bins=10, num_classes=self.num_classes)

        self.setup_model()
        self.setup_tensorboard()

    def setup_tensorboard(self):

        log_dir = (
            f"../../tensorboard/optimizers/Dataset_type: {self.dataset_type}/"
            f"Dataset_name: {self.dataset_name}/"
            f"Model: {self.layer_type}/"
            f"Optimizer: {self.optimizer_type}/" 
            f"N_Layers: {self.num_layers}/"
            f"lr: {self.lr}/"
            f"wd: {self.weight_decay}/"
            f"exp: {self.experiment_number}/ "
            + (f"test_samples: {self.test_samples}/train_samples: {self.train_samples}/" 
            if self.optimizer_type == "ivon" else "")
        )

        self.writer = SummaryWriter(log_dir)

        #print(f"opti {self.optimizer_type}")

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
            print(self.data)
            self.optimizer = IVON(
                self.model.parameters(), lr=self.lr, ess=self.num_samples
            )

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
            train_acc, val_acc, test_acc, ece, brier, entropy, nll = self._evaluate(self.data, self.edge_index)

            #print(f"step: loss {train_loss}")
            self.writer.add_scalar("Training loss", train_loss, epoch)
            self.writer.add_scalar("Training accuracy", train_acc, epoch)
            self.writer.add_scalar("Validation accuracy", val_acc, epoch)
            self.writer.add_scalar("Test accuracy", test_acc, epoch)
            self.writer.add_scalar("Test ECE", ece, epoch)
            self.writer.add_scalar("Brier Score", brier, epoch)
            self.writer.add_scalar("Mean entropy", entropy, epoch)
            self.writer.add_scalar("NLL", nll, epoch)

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

        ece = self.test_ece(sm[data.test_mask], data.y[data.test_mask])
        brier = brier_score(sm[data.test_mask], data.y[data.test_mask])

        entropy = -(sm * torch.log(sm + 1e-12)).sum(dim=1)
        mean_entropy = entropy[data.test_mask].mean().item()

        nll = F.nll_loss(log_probs[data.test_mask], data.y[data.test_mask]).item()

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        return train_acc.item(), val_acc.item(), test_acc.item(), ece, brier, mean_entropy, nll



    



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
