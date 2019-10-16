To train with the TVO objective, modify the data paths in [`data.py`](https://github.com/vmasrani/tvo/blob/master/discrete_vae/data.py#L14-L17) and run
```
python run.py --train-mode thermo
```
Running this will create a folder `save/` and save models and checkpoints there.

For help, run 
```
python run.py --help
```

The TVO loss is computed in [`get_thermo_loss_from_log_weight_log_p_log_q` in `losses.py`](https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py#L248-L316).

The main training loop is in [`train_thermo` in `train.py`](https://github.com/vmasrani/tvo/blob/master/discrete_vae/train.py#L340-L364).

Discrete VAE models are in [`models.py`](https://github.com/vmasrani/tvo/blob/master/discrete_vae/models.py) and plotting scripts for the figure in the paper are in [`plot.py`](https://github.com/vmasrani/tvo/blob/master/discrete_vae/plot.py)
