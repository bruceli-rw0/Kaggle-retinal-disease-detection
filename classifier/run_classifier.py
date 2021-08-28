import os
import glob
import pickle
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
try:
    from .dataset import RetinalDataset
    from .metrics import RetinalF1Metrics
    from .models import RetinalModel
except:
    from dataset import RetinalDataset
    from metrics import RetinalF1Metrics
    from models import RetinalModel

logger = logging.getLogger(__name__)

def train(args, dataloader, model, criterion, optimizer, metrics) -> None:
    running_loss = 0
    model = model.train()
    for images, labels in tqdm(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        metrics.update_value(
            labels.to('cpu').numpy(), 
            nn.Sigmoid()(logits).detach().to('cpu').numpy()
        )

    loss = images = labels = None
    metrics.update_loss(running_loss / len(dataloader))
    return running_loss / len(dataloader)

def evaluate(args, dataloader, model, criterion, metrics=None) -> None:
    running_loss = 0
    model = model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            metrics.update_value(
                labels.to('cpu').numpy(), 
                nn.Sigmoid()(logits).detach().to('cpu').numpy()
            )
    loss = images = labels = None
    metrics.update_loss(running_loss / len(dataloader))
    return running_loss / len(dataloader)

def save_plot(train_metrics, dev_metrics, save_idx):
    train_f1, train_loss =  train_metrics.running_f1, train_metrics.running_loss
    dev_f1, dev_loss =  dev_metrics.running_f1, dev_metrics.running_loss
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

    axes[0].plot(train_loss)
    axes[0].plot(dev_loss)
    axes[0].legend(["train", "dev"])
    axes[0].set_ylim([0,0.6])
    axes[0].set_title("Loss")

    axes[1].plot(np.mean(train_f1, axis=1))
    axes[1].plot(np.mean(dev_f1, axis=1))
    axes[1].legend(["train", "dev"])
    axes[1].set_ylim([0.3,1])
    axes[1].set_title("F1")

    fig.savefig(os.path.join("_plots", f"{save_idx}.jpg"), bbox_inches='tight')

def setup_logging(args):
    logfiles = glob.glob("_loggings/*.log")
    if args.next_idx is None:
        next_idx = 0
        if logfiles:
            next_idx = max([int(f.split("/")[-1].split(".")[0]) for f in logfiles]) + 1
        args.next_idx = next_idx

    handlers = list()
    if args.save_log:
        handlers.append(logging.FileHandler(filename=os.path.join("_loggings", f"{args.next_idx}.log"), mode='w'))
    if args.verbose_log:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )
    logger.info(f"Index: {args.next_idx}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Use pretrained: {args.pretrained}")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Momentum: {args.momentum}")
    logger.info(f"Batch size: {args.batch_size}")

def initalization(args) -> None:
    torch.manual_seed(args.random_state)

    if "_loggings" not in os.listdir():
        os.mkdir("_loggings")
    if "_checkpoints" not in os.listdir():
        os.mkdir("_checkpoints")
    if "_stats" not in os.listdir():
        os.mkdir("_stats")
    if "_plots" not in os.listdir():
        os.mkdir("_plots")
    setup_logging(args)

    # load data
    train_data = RetinalDataset(
        args.train_folder, 
        args.train_label_file, 
        training=True, 
        experiment=args.experiment,
        experiment_batch=args.experiment_batch
    )
    dev_data = RetinalDataset(
        [args.dev_folder], 
        [args.dev_label_file], 
        training=False, 
        experiment=args.experiment,
        experiment_batch=args.experiment_batch
    )
    test_data = RetinalDataset(
        [args.test_folder], 
        [args.test_label_file], 
        training=False, 
        experiment=args.experiment,
        experiment_batch=args.experiment_batch
    )

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    devloader = DataLoader(dev_data, batch_size=1)
    testloader = DataLoader(test_data, batch_size=1)

    # initialize model
    model = RetinalModel(args.model, args.pretrained, args.num_classes).to(args.device)
    # set up training hyperparameters
    optimizer = getattr(optim, args.optimizer)(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(args.momentum, 0.99),
        weight_decay=args.lr_decay
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=5, 
        verbose=True
    )
    criterion = nn.BCEWithLogitsLoss()

    train_metrics = RetinalF1Metrics(args.num_classes, args.train_label_file[0], data_type="train")
    dev_metrics = RetinalF1Metrics(args.num_classes, args.dev_label_file, data_type="dev")
    test_metrics = RetinalF1Metrics(args.num_classes, args.test_label_file, data_type="test")
    best_dev_f1 = -1

    if args.do_train:
        for e in range(args.epochs):
            logger.info(f"\nEpoch: {e+1}/{args.epochs} ...")
            loss = train(args, trainloader, model, criterion, optimizer, train_metrics)
            train_metrics.new_epoch(logger)
            logger.info(f"Epoch train loss: {loss:.4f}")

            if args.in_training_evaluation:
                loss = evaluate(args, devloader, model, criterion, dev_metrics)
                dev_metrics.new_epoch(logger)
                logger.info(f"Epoch dev loss: {loss:.4f}")

                if dev_metrics.get_current() > best_dev_f1:
                    # update learning rate scheduler
                    lr_scheduler.step(dev_metrics.get_current())
                    best_dev_f1 = dev_metrics.get_current()
                    if args.save_model:
                        # save model when dev f1 score improves
                        torch.save(model.state_dict(), os.path.join("_checkpoints", f"{args.next_idx}.pth"))
            
            save_plot(train_metrics, dev_metrics, args.next_idx)

        if args.save_stats:
            # save training stats
            with open(os.path.join("_stats", f"{args.next_idx}.train.stats"), "wb") as f:
                pickle.dump([train_metrics.running_f1, train_metrics.running_loss], f)
            if args.in_training_evaluation:
                # save development stats
                with open(os.path.join("_stats", f"{args.next_idx}.dev.stats"), "wb") as f:
                    pickle.dump([dev_metrics.running_f1, dev_metrics.running_loss], f)

    if args.do_test:
        if args.do_train:
            # load the best model from training
            model.load_state_dict(torch.load(
                os.path.join("_checkpoints", f"{args.next_idx}.pth"), 
                map_location=torch.device(args.device)
            ))

        elif args.checkpoint_idx is not None:
            # load specific checkpoints
            model.load_state_dict(torch.load(
                os.path.join("_checkpoints", f"{args.checkpoint_idx}.pth"), 
                map_location=torch.device(args.device)
            ))

        else:
            logger.info("No pretrained model defined for testing.")
            return

        evaluate(args, testloader, model, criterion, test_metrics)
        test_metrics.new_epoch(logger)