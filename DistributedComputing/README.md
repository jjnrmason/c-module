# Distributed Computing

## Number file processing with MPI

Written in C and uses the mpi library. Program can run with parallel processes to parse a file of numbers seperated by comma and newline before processing the following information:

1. Total count
2. Average
3. Max
4. Min

### Compile Command:
```
mpicc mpi-nums-processor.c
```

### Run Command:
```
mpiexec ./a.out

or for multiprocessing

mpiexec -n 4 -oversubscribe ./a.out
```