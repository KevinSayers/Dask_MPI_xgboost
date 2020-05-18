# XGBoost with Dask-mpi example

Using Dask-mpi/Dask-xgboost to run a sample Iris model. 

General steps:
* Create Dask client using Dask-MPI, this creates the Dask scheduler and Dask workers on MPI ranks
* Setup model
* Use Dask-XGBoost to run training: this splits the training across the Dask workers via the client


## Setup

```shell
conda env create -f environment.yml
```

```shell
conda activate xgboost_env
```

## Run

```shell
ml intel-mpi/2018.0.3
mpiexec -n 4 python xgb_test.py
```

## References
1. [A New, Official Dask API for XGBoost](https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7)
2. [A Beginner’s guide to XGBoost](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)
