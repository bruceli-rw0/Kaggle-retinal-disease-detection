import argparse
import warnings
try:
    from .run_classifier import initalization
    from .label_propagation import label_propagation
except:
    from run_classifier import initalization
    from label_propagation import label_propagation

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Process settings.')

    parser.add_argument("--train_folder", nargs='+', type=str, default=["datasets/train-train"])
    parser.add_argument("--train_label_file", nargs='+', type=str, default=["datasets/train-train.csv"])
    parser.add_argument("--dev_folder", type=str, default="datasets/train-dev")
    parser.add_argument("--dev_label_file", type=str, default="datasets/train-dev.csv")
    parser.add_argument("--test_folder", type=str, default="datasets/train-test")
    parser.add_argument("--test_label_file", type=str, default="datasets/train-test.csv")
    parser.add_argument("--unlabel_folder", type=str, default="datasets/test")
    # parser.add_argument("--test_pred_file", type=str, default="datasets/train-test-pred.csv")

    parser.add_argument("--verbose_log", action='store_true')
    parser.add_argument("--save_log", action='store_true')
    parser.add_argument("--save_stats", action='store_true')
    parser.add_argument("--save_model", action='store_true')
    
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--in_training_evaluation", action='store_true')

    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_decay", type=float, default=1e-5)

    parser.add_argument("--next_idx", type=int, default=None)
    parser.add_argument("--checkpoint_idx", type=int, default=None)

    parser.add_argument("--label_propagation", action='store_true')
    parser.add_argument("--kernel", type=str, default='rbf')
    # kernel_setting will be gamma for rbf or n_neighbors for knn
    parser.add_argument("--kernel_setting", nargs='+', type=float, default=[0.02])

    parser.add_argument("--experiment", action='store_true')
    parser.add_argument("--experiment_batch", type=int, default=3)
    
    args = parser.parse_args()

    if args.do_train or args.do_test:
        initalization(args)
    if args.label_propagation:
        label_propagation(args)

if __name__=="__main__":
    main()