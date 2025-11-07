import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from .model import GraphLevelGNN 
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb

def train_graph_classifier(model_name, train_set, val_set, test_set, dataset,
                            CHECKPOINT_PATH, AVAIL_GPUS, epochs=100, **model_kwargs):
    pl.seed_everything(42)
    train_loader = DataLoader(train_set, batch_size=8)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)
    c_out = dataset.num_classes

    wandb_logger = WandbLogger()

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(dirpath=root_dir, save_weights_only=True, mode="max", monitor="val_auc", filename=f'GraphLevel{model_name}'),
            LearningRateMonitor(logging_interval="step")],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        max_epochs=epochs,
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=root_dir), wandb_logger],
    )  # 0 because epoch size is 1

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "GraphLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN(
            model_name=model_name, num_PPI_type=dataset.num_node_features, c_out=c_out, **model_kwargs
        )
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = GraphLevelGNN(
            model_name=model_name, num_PPI_type=dataset.num_node_features, c_out=c_out, **model_kwargs
        )
        trainer.fit(model, train_loader, val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # # Test best model on the test set
    test_result = trainer.test(model, dataloaders=test_loader , verbose=False)
    batch = next(iter(test_loader))
    batch = batch.to(model.device)
    _, train_acc, *_ = model.forward(batch, mode="train")
    _, val_acc, *_ = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result, trainer

def train_graph_classifier_kfold(model_name, train_set, val_set, dataset,
                            CHECKPOINT_PATH, AVAIL_GPUS, epochs=100, batch_size=256,
                            **model_kwargs):
    pl.seed_everything(42)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    c_out = dataset.num_classes

    wandb_logger = WandbLogger()

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(dirpath=root_dir, save_weights_only=True, mode="max", monitor="val_auc", filename=f'GraphLevel{model_name}'),
            LearningRateMonitor(logging_interval="epoch")],
        accelerator="gpu",
        devices='auto',
        max_epochs=epochs,
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=root_dir), wandb_logger],
    )  # 0 because epoch size is 1

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "GraphLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN(
            model_name=model_name, num_PPI_type=dataset.num_node_features, c_out=c_out, **model_kwargs
        )
    else:
        pl.seed_everything()
        model = GraphLevelGNN(
            model_name=model_name, num_PPI_type=dataset.num_node_features, c_out=c_out, **model_kwargs
        )
        trainer.fit(model, train_loader, val_loader)
