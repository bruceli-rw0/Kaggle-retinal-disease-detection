# Retinal Disease Detection

Kaggle link: https://www.kaggle.com/c/vietai-advance-course-retinal-disease-detection/overview

## Installation
Run the following command in the console to set up virtual environment and install required packages.
```bash
make setup
```

## Usage
Train and test
```bash
./env/bin/python3 -m classifier \
		--verbose_log \
		--save_log \
		--save_stats \
		--save_model \
		--do_train \
		--do_test \
		--in_training_evaluation \
		--model resnest50 \
		--optimizer Adam \
		--learning_rate 0.001 \
		--lr_decay 0.00001 \
        --batch_size 111 \
		--epochs 100
```

Label propagation
```bash
./env/bin/python3 -m classifier \
        --verbose_log \
        --save_log \
        --label_propagation \
        --model resnest50 \
        --checkpoint_idx 2
```

### Caveat
* `resnest50` + `batch_size=128` will run out of cuda memory in colab
