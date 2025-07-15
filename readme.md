# 部署

python=3.10

```
pip install -r requirements.txt
```

# 测试

在m3gnet的外面，和m3gnet同级进行测试

```
python -m m3gnet.graph.tests.test_graph_and_batch
python -m m3gnet.layers.tests.test_layers
python -m m3gnet.models.tests.test_models
python -m m3gnet.train.tests.test_trainer
```

或者在内部使用

```
python test.py
```

# 训练与评估

```
python run_train.py
python predict.py saved_models/property_predictor/best_model data/cif_file/
python predict.py saved_models/property_predictor/best_model data/cif_file/ --no-plot
```
