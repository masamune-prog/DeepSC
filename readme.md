# DiffusionSC


This is the implementation of DiffusionSC

## Requirements
+ See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.


```
## Preprocess
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess_text.py
```

## Train
```shell
PYTHONUNBUFFERED=1 uv run -u python diffusion_original.py 
```


## Evaluation
```shell
PYTHONUNBUFFERED=1 uv run python -u diffusion_perf_ori.py --bert-score --channel Rayleigh --sampler ddim --epoch-seeds 10 20 30 40 50 --epochs 5
```

