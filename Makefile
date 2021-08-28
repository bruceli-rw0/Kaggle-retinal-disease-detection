.PHONY: all
all: experiment

.PHONY: setup
setup:
	python3 -m venv env
	./env/bin/python3 -m pip install --upgrade pip
	./env/bin/python3 -m pip install -r requirements.txt

.PHONY: experiment
experiment:
	clear
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
		--epochs 3 \
		--batch_size 3 \
		--experiment_batch 9 \
		--experiment

.PHONY: run
run:
	clear
	./env/bin/python3 -m classifier \
		--verbose_log \
		--save_log \
		--save_stats \
		--save_model \
		--do_train \
		--in_training_evaluation \
		--model resnet50 \
		--pretrained \
		--optimizer Adam \
		--device cuda \
		--learning_rate 0.001 \
		--lr_decay 0.00001 \
		--batch_size 128 \
		--epochs 75

.PHONY: semi
semi:
	clear
	./env/bin/python3 -m classifier \
		--label_propagation \
		--model resnest50 \
		--checkpoint_idx 2
		# --batch_size 3 \
		# --experiment_batch 9 \
		# --experiment

.PHONY: full
full:
	clear
	./env/bin/python3 -m classifier \
	--verbose_log \
		--save_log \
		--save_stats \
		--save_model \
		--do_train \
		--do_test \
		--in_training_evaluation \
		--device cpu \
		--model resnest50 \
		--optimizer Adam \
		--learning_rate 0.001 \
		--lr_decay 0.00001 \
		--epochs 3 \
		--batch_size 3 \
		--experiment_batch 9 \
		--experiment \
		--label_propagation

.PHONY: clean
clean:
	rm -rf _checkpoints
	rm -rf _loggings
	rm -rf _loggings-semi
	rm -rf _stats
	rm -rf _plots

.PHONY: submit
submit:
	rm submission.zip
	zip -r submission.zip \
		classifier/* \
		datasets/train-dev.csv \
		datasets/train-test.csv \
		datasets/train-train.csv \
		notebooks/access-data.ipynb \
		notebooks/training-log.ipynb \
		report.md \
		media/* \
		-x classifier/__pycache__/*