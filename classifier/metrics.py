import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def multi_label_f1(y_gt, y_pred):
    """ Calculate F1 for each class

    Parameters
    ----------
    y_gt: torch.Tensor - groundtruth
    y_pred: torch.Tensor - prediction

    Returns
    -------
    list - F1 of each class

    This function is adapted from https://www.kaggle.com/hmchuong/pytorch-baseline-model
    """
    f1_out = []
    gt_np = y_gt
    pred_np = (y_pred > 0.5) * 1.0
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        f1_out.append(f1_score(gt_np[:, i], pred_np[:, i]))
    return f1_out

class RetinalF1Metrics:
    def __init__(self, num_classes: int, label_file: int, data_type: str):
        self.label_names = pd.read_csv(label_file).columns.tolist()[1:]
        assert num_classes == len(self.label_names)

        self.type = data_type
        self.best_f1 = 0
        self.current_f1 = 0
        self.running_f1 = list()
        self.running_loss = list()
        self.epoch_pred = None
        self.epoch_label = None

    def update_loss(self, loss):
        self.running_loss.append(loss)
    
    def update_value(self, label: np.array, pred: np.array):
        if self.epoch_pred is None:
            self.epoch_pred = pred.copy()
            self.epoch_label = label.copy()
        else:
            self.epoch_pred = np.concatenate([self.epoch_pred, pred], axis=0)
            self.epoch_label = np.concatenate([self.epoch_label, label], axis=0)

    def new_epoch(self, logger):
        epoch_f1 = multi_label_f1(self.epoch_label, self.epoch_pred)
        self.running_f1.append(epoch_f1)
        self.current_f1 = np.mean(epoch_f1)

        logger.info(f"Epoch {self.type} f1 score")
        for name, f1 in zip(self.label_names, epoch_f1):
            logger.info(f"\t {name}: {f1:.4f}")
        logger.info(f"Average: {np.mean(epoch_f1):.4f}")
        if np.mean(epoch_f1) > self.best_f1:
            self.best_f1 = np.mean(epoch_f1)
        logger.info(f"Best: {self.best_f1:.4f}")
        
        # reset epoch stats
        self.epoch_pred = None
        self.epoch_label = None

    def get_best(self):
        return self.best_f1

    def get_current(self):
        return self.current_f1