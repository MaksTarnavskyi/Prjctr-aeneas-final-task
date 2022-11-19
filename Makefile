build:
	docker build -f Dockerfile -t aeneas:latest .

run_dev: build
	docker run -it -v ${PWD}:/main aeneas:latest /bin/bash


preprocess_data:
	python3 utils/preprocess_data.py --source_file data/train.src --labels_file data/train.lbl --output_file data_processed/train

split_train_val_data:
	python3 utils/split_train_val.py --input_filename data_processed/train --output_train_filename data_processed/train_98 --output_val_filename data_processed/val_02

preprocess_data_all:
	preprocess_data
	split_train_val_data

train_default:
	python3 train.py --transformer_name distilroberta-base --train_set data_processed/train_98 --dev_set data_processed/val_02 --model_dir models/distilroberta-base

predict_file_default:
	python3 predict.py --transformer_name distilroberta-base --model_dir models/distilroberta-base/distilroberta-base --source_file data/val.src --output_file data/val.pred.lbl.distilroberta-base

build_fast_api:
	docker build -f Dockerfile -t app-fastapi:latest --target app-fastapi .  

run_fast_api: build_fast_api
	docker run -it -p 8080:8080 app-fastapi:latest
