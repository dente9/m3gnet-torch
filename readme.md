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

# 训练与评估

```
python run_train.py
python run_evaluate.py --model-dir saved_models/property_predictor --model-type best
```
