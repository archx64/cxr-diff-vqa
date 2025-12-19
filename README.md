# DRIFT

How to train?

Install the dependencies from req.txt and execute this command.

```bash
CUDA_VISIBLE_DEVICES="Index of your GPU" nohup python3 -u -m src.train --config configs/clinicalbert_resnet.yaml > nohup_train.log 2>&1 &
```
