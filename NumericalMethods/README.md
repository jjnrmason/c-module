# Numerical Methods

## Linear regression

Written in C. Program can parse a given file containing the points A and B before applying them to the linear regression formula.

### Compile Command:
```
cc calc-linear-regression.c
```

### Run Command:
```
./a.out
```

## Leibniz Formula

Written in C and uses pthreads. Program can calculate the Leibniz Formula to a given number of iterations by args. It can be sped up using threads.

### Compile Command:
```
cc calc-leibniz-formula.c -lm -pthread
```

### Run Command:
```
./a.out 2 20
```

## Prime number file search with pthread

Written in C and uses pthreads. Program can read a file provided and output all prime numbers found alongside the total to a file.

### Compile Command:
```
cc find-primes.c -pthread
```

### Run Command:
```
./a.out 2
```

## Image box blur with lodepng.c

Written in C and uses pthreads & [lodepng](https://lodev.org/lodepng/). Program can blur a parsed image and save the result.

### Compile Command:
```
cc image-box-blur.c lodepng.c -lm -pthread
```

### Run Command:
```
./a.out Images/testpepe.png Images/result.png 5
```
