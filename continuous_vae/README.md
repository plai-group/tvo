This uses [Sacred's](https://sacred.readthedocs.io/en/stable/command_line.html) command line interface. To see Sacred's options run

```
python run.py --help
```

To see tunable hyperparameters

```
python run.py print_config
```

which can be set using `with`:

```
python run.py with loss='thermo' S=10 seed=2 epochs=10 train_only=True -p
```

To save data to the filesystem, add a Sacred [FileStorageObserver](https://sacred.readthedocs.io/en/stable/observers.html)

```
python run.py with loss='thermo' S=10 seed=2 epochs=10 train_only=True -p -F ./runs
```

The commands to reproduce the results in the paper are in `make_data_for_paper.sh`. Note that for 5000 epochs, each run takes ~20 hrs with a gpu.  This will save training output to `./runs`. Code to read from `./runs` and reproduce figures 6 and 7 in the paper is in `notebooks/plots.ipydb`.

The TVO loss is computed in [`get_thermo_loss_from_log_weight_log_p_log_q` in `losses.py`](https://github.com/vmasrani/tvo/blob/master/continuous_vae/losses.py#L149-L202). This function is identical to the one found in the discrete_vae directory.

The main training loop is in [`train` in `run.py`](https://github.com/vmasrani/tvo/blob/master/continuous_vae/run.py#138).

