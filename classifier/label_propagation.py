import os
import glob
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import f1_score
try:
    from .dataset import RetinalDataset
    from .models import RetinalModel
except:
    from dataset import RetinalDataset
    from models import RetinalModel

logger = logging.getLogger(__name__)

def feature_extraction(args, dataloader, load_index):
    X = None
    model = RetinalModel(args.model, args.pretrained, args.num_classes)
    model.load_state_dict(torch.load(
        os.path.join("_checkpoints", f"{load_index}.pth"), 
        map_location=torch.device(args.device)
    ))
    model.to(args.device)
    model = model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(args.device)
            features = model.extract_features(images)
            if X is None:
                X = features
            else:
                X = torch.cat((X, features), 0)
    return X.detach().to('cpu').numpy()

def label_propagation(args):
    if args.next_idx is None and args.checkpoint_idx is None:
        print("No pre-trained model indicated for feature extraction")
        return

    load_index = args.next_idx if args.next_idx is not None else args.checkpoint_idx
    assert f"{load_index}.pth" in os.listdir("_checkpoints")

    if "_loggings-semi" not in os.listdir():
        os.mkdir("_loggings-semi")

    handlers = list()
    if args.save_log:
        handlers.append(logging.FileHandler(filename=os.path.join("_loggings-semi", f"{load_index}.log"), mode='w'))
    if args.verbose_log:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )

    label_data = RetinalDataset(
        args.train_folder, 
        label_files=None,
        training=False, 
        experiment=args.experiment,
        experiment_batch=args.experiment_batch
    )
    unlabel_data = RetinalDataset(
        [args.unlabel_folder], 
        label_files=None,
        training=False, 
        experiment=args.experiment,
        experiment_batch=args.experiment_batch
    )
    labelloader = DataLoader(label_data, batch_size=1)
    unlabelloader = DataLoader(unlabel_data, batch_size=1)
    unlabel_path = sorted(glob.glob(os.path.join(args.unlabel_folder, "*.jpg")))

    Y_label = pd.read_csv(args.train_label_file[0]).iloc[:,1:].to_numpy().T
    if args.experiment:
        Y_label = Y_label[:,:args.experiment_batch]
        unlabel_path = unlabel_path[:args.experiment_batch]

    # initialize output dataframe
    columns = pd.read_csv(args.dev_label_file).columns
    unlabel_df = pd.DataFrame(columns=columns)
    unlabel_df['filename'] = [path.split("/")[-1] for path in unlabel_path]

    # extract dense layer representation of images
    X_label = feature_extraction(args, labelloader, load_index)
    X_unlabel = feature_extraction(args, unlabelloader, load_index)
    X = np.concatenate([X_label, X_unlabel], axis=0)

    logger.info("Begin label propagation ...")
    best_f1 = -1
    for kernel_set in args.kernel_setting:
        current_df = unlabel_df.copy()
        label_f1 = []

        # perform label propagation for each class
        for i, y_label in enumerate(Y_label):
            y_unlabel = np.ones((X_unlabel.shape[0],)) * -1
            y = np.concatenate([y_label, y_unlabel], axis=0)
            y = y.astype(np.int64)

            model = LabelPropagation(
                kernel=args.kernel,
                gamma=kernel_set,
                n_neighbors=kernel_set,
                n_jobs=-1
            )
            model.fit(X, y)
            current_df.iloc[:,i+1] = model.transduction_[-X_unlabel.shape[0]:]
            y_label_pred = model.predict(X_label)
            label_f1.append(f1_score(y_label, y_label_pred))

        logger.info(f"Setting: {kernel_set} \t Average F1: {np.mean(label_f1):.4f}")

        if np.mean(label_f1) > best_f1:
            best_f1 = np.mean(label_f1)

            # post-processing
            # make sure normal=0 if at least one of the other label is 1
            # set normal=1 if all label is 0
            arr = current_df.iloc[:,1:].to_numpy()
            arr[:,-1][arr[:,:-1].sum(axis=1) == 0] = 1
            arr[:,-1][arr[:,:-1].sum(axis=1) != 0] = 0
            current_df.iloc[:,1:] = arr

            current_df.to_csv(os.path.join("datasets", "test-label.csv"), index=False)