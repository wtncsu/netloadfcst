# DoE project (fuzzy decision trees)

## Cloning

Use the following command

```
git clone https://github.ncsu.edu/Mercurial-Potato/doe-project.git
```

## Dependencies
You will need to install `GNU make` and `python3` and the following additional 
python packages:

Use `pip install` 

1. `numpy`
1. `scipy`
1. `matplotlib`
1. `tqdm`

## Generating Output

`GNU Make` will manage dependency so that when the config file for
the model is changed, the model is re-built.

Fit and predict all models using the following command.

```
# fit one model at a time
make all

# fits models in parallel using all available processors 
make -j$(nproc) all
```
Note that in parallel mode, the output messages will be garbled.

The following targets perform a partial build

```
make all-dt    # Decision tree
make all-fdt   # Fuzzy decision tree
make all-bdt   # Boosted decision tree
make all-bfdt  # Boosted fuzzy decision tree
```

## Plotting targets

The following targets will open a figure window.

```
make plot-dt-test-S1
make plot-fdt-train-S3-AT
make plot-bfdt-test-S3-AT

... so on
```
