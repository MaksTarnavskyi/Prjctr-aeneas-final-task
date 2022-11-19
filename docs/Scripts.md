Scripts how to train the Aeneas model and generate prediction
=================
### Install requirements

```python
pip install -r requirements.txt
```

```bash
mkdir models cache data_processed
```

### Prepare data for training model

```python
python3 utils/preprocess_data.py --source_file data/train.src --labels_file data/train.lbl --output_file data_processed/train
```

```python
python3 utils/split_train_val.py --input_filename data_processed/train --output_train_filename data_processed/train_98 --output_val_filename data_processed/val_02
```

### Train model
```python
python3 train.py --transformer_name distilroberta-base --train_set data_processed/train_98 --dev_set data_processed/val_02 --model_dir models/distilroberta-base
```

### Predict labels using trained model

distilroberta-base
```python
python3 predict.py --transformer_name distilroberta-base --model_dir models/distilroberta-base/distilroberta-base --source_file data/val.src --output_file data/val.pred.lbl.distilroberta-base
```

distilbert-base-cased
```python
python3 predict.py --transformer_name distilbert-base-cased --model_dir models/distilbert-base-cased --source_file data/val.src --output_file data/val.pred.lbl.distilbert-base-cased
```

albert-base-v2
```python
python3 predict.py --transformer_name albert-base-v2 --model_dir models/albert-base-v2 --source_file data/val.src --output_file data/val.pred.lbl.albert-base-v2
```

### Ensemble
```python
python utils/ensemble.py --input_files data/val.pred.lbl.distilroberta-base data/val.pred.lbl.distilbert-base-cased data/val.pred.lbl.albert-base-v2 --output_file data/val.pred.lbl.ensemble
```

### Evaluation
```python
python3 eval.py data/val.pred.lbl.distilroberta-base
```