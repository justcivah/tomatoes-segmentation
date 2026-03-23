import torch
import os
from datetime import datetime
import mlflow
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryJaccardIndex,
)

class SimpleTrainer():
    def __init__(self, model, train_loader, valid_loader, optimizer, loss_fn, epochs, experiment_name, run_name=None, checkpoint_path=None, metrics_threshold=0.5, hparams=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.experiment_name = experiment_name
        self.best_valid_loss = float('inf')
        self.metrics_threshold = metrics_threshold
        self.hparams = hparams or {}

        # infer device from model parameters
        self.device = next(model.parameters()).device

        if checkpoint_path is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            checkpoint_path = os.path.join('models', timestamp) + '/'

        os.makedirs(checkpoint_path, exist_ok=True)
        self.checkpoint_path = checkpoint_path

        mlflow.set_experiment(experiment_name)
        self.run_name = run_name


    def fit(self):
        with mlflow.start_run(run_name=self.run_name) as run:
            self._log_hparams()
            print(f"mlflow run id: {run.info.run_id}")

            for epoch in range(self.epochs):
                # training
                self.model.train()
                train_loss = self._train_one_epoch()

                # validation
                self.model.eval()
                metrics = self.evaluate_metrics(self.valid_loader)

                self._print_metrics(epoch, train_loss, metrics)
                self._log_epoch(epoch, train_loss, metrics)

                val_loss = metrics["loss"]
                if val_loss < self.best_valid_loss:
                    self.best_valid_loss = val_loss
                    self._save_checkpoint(epoch, val_loss, self.checkpoint_path + 'best_model.pt')
                    print(f'new best model saved with valid loss {val_loss:.4f}')

            self._save_checkpoint(epoch, val_loss, self.checkpoint_path + 'last_model.pt')


    def evaluate_metrics(self, dataloader):
        metrics = self._build_metrics()
        running_loss = 0.0
        total_batches = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss    = self.loss_fn(outputs, targets)
                running_loss += loss.item()
                total_batches += 1

                # converting into probabilities and then flattening to 1D
                probs  = torch.sigmoid(outputs).view(-1)
                labels = targets.long().view(-1)

                preds = (probs >= self.metrics_threshold).long()
                metrics["iou"].update(preds, labels)
                metrics["dice"].update(preds, labels)
                metrics["precision"].update(preds, labels)
                metrics["recall"].update(preds, labels)

                metrics["auroc"].update(probs, labels)
                metrics["avg_precision"].update(probs, labels)

        return {
            "loss":          running_loss / total_batches,
            "iou":           metrics["iou"].compute().item(),
            "dice":          metrics["dice"].compute().item(),
            "precision":     metrics["precision"].compute().item(),
            "recall":        metrics["recall"].compute().item(),
            "auroc":         metrics["auroc"].compute().item(),
            "avg_precision": metrics["avg_precision"].compute().item(),
        }
    

    def _build_metrics(self):
        return {
            "iou":           BinaryJaccardIndex(threshold=self.metrics_threshold).to(self.device),
            "dice":          BinaryF1Score(threshold=self.metrics_threshold).to(self.device),
            "precision":     BinaryPrecision(threshold=self.metrics_threshold).to(self.device),
            "recall":        BinaryRecall(threshold=self.metrics_threshold).to(self.device),
            "auroc":         BinaryAUROC().to(self.device),
            "avg_precision": BinaryAveragePrecision().to(self.device),
        }

    
    def _train_one_epoch(self):
        running_loss = 0.
        total_batches = 0

        for inputs, targets in self.train_loader:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
        
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / total_batches

        return epoch_loss


    def _log_hparams(self):
        base = {
            "epochs":            self.epochs,
            "metrics_threshold": self.metrics_threshold,
            "device":            str(self.device),
            "optimizer":         self.optimizer.__class__.__name__,
        }

        mlflow.log_params({**base, **self.hparams})
    

    def _log_epoch(self, epoch, train_loss, metrics):
        mlflow.log_metrics(
            {
                "train_loss":   train_loss,
                "valid_loss":   metrics["loss"],
                "valid_iou":    metrics["iou"],
                "valid_dice":   metrics["dice"],
                "valid_prec":   metrics["precision"],
                "valid_recall": metrics["recall"],
                "valid_auroc":  metrics["auroc"],
                "valid_ap":     metrics["avg_precision"],
            },
            step=epoch,
        )


    def _save_checkpoint(self, epoch, val_loss, path):
        checkpoint = {
            'epoch':            epoch,
            'val_loss':         val_loss,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }

        torch.save(checkpoint, path)


    def _print_metrics(self, epoch, train_loss, metrics):
        print(
            f"epoch {epoch:>3} | "
            f"train loss: {train_loss:.4f} | "
            f"valid loss: {metrics['loss']:.4f} | "
            f"iou: {metrics['iou']:.4f} | "
            f"dice: {metrics['dice']:.4f} | "
            f"precision: {metrics['precision']:.4f} | "
            f"recall: {metrics['recall']:.4f} | "
            f"auroc: {metrics['auroc']:.4f} | "
            f"ap: {metrics['avg_precision']:.4f}"
        )