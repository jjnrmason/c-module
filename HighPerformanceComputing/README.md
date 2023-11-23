# High Performance Computing

## Password crack

Written in C and uses pthreads & crypt. Program can crack a SHA512 encrypted string in the format 'AA00' (a capital letter, a capital letter, a number between 0-99).

### Compile Command:
```
cc password-crack.c -lcrypt -pthread
```

### Run Command:
```
./a.out '$6$AS$IKP6957BKgwlCeaHhrI3h86WuUSwWUfXlSBnbR/dfjbnXXO7E4PciSyvOb5SVUgTRvaB.tX4SuAtj1qgyo/AA.' 2
```

## Matrix Multiplication

Written in C and uses pthreads. Program can parse a file in a given format such as [input-matrices.txt](./HighPerformanceComputing/MatrixMultiplication/input-matrices.txt) that contains matrices with their size and values. Each matrix is seperates by a newline character.

### Compile Command:
```
cc matrix-multiplication.c -pthread
```

### Run Command:
```
./a.out 'input-matrices.txt' 2 'output-matrices.txt'
```

## Password crack (CUDA)

Written in CUDA. Program can crack an encrypted string in the format 'AA00' (a capital letter, a capital letter, a number between 0-99).

### Compile Command:
```
nvcc password-crack.c
```

### Run Command:
```
./a.out 'lhkmgi3153'
```

## Box blur (CUDA)

Written in CUDA. Program can crack blur a png image.

### Compile Command:
```
nvcc box-blur.cu lodepng.cpp
```

### Run Command:
```
./a.out pepe.png pepe-out.png
```