# Fuzzy Model for Timeseries Forecasting

[Google Drive](https://drive.google.com/drive/folders/1czbv8byjEYlFnsxwmx0JQVnAbyRjRSjH?usp=sharing)

## Unit Tests

Debugging faulty code is extremely time-consuming especially when the defects
are in one of the submodules. With a short deadline you would think adding unit
tests would delay the project but due to the higher code quality, much less time
is wasted on debugging.

Test cases are automatically tested against subsequent changes to the module
(instead of manual code inspection). This can avoid repeating work.


Run all unit tests with the following command in the project folder

```
python3 -m unittest discover .
```

## Fitting Models

1. Change the parameters in the `.ini` config files
1. Run the following command. Dispatch jobs to fill all processor cores.

```
make -j$(nproc)
```

All models with modified data or config files will be re-evaluated. The output
is in folder `output`.
