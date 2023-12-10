#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>

typedef struct {
    int start;
    int end;
    int rowsA;
    int colsA;
    int rowsB;
    int colsB;
    double **matrixA;
    double **matrixB;
    double **resultMatrix;
} ThreadData;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *multiplyMatrices(void *threadArgs);
void printMatrix(double **matrix, int rows, int cols);
bool canMultiplyMatrices(int matrixACols, int matrixBRows);

int main(int argc, char *argv[]) {
    char *resultsFilePath;
    if (argc < 3) {
        printf("Program ran with the incorrect number of args.\n");
        printf("./a.out {{MatrixFilePath}} {{NumberOfThreads}}\n");
        return 1;
    } else if (argc == 3) {
        printf("If you want to specify your output file do so in the forth arg.\n");
        printf("./a.out {{MatrixFilePath}} {{NumberOfThreads}} {{ResultFilePath}}\n");
        resultsFilePath = "output-matrices.txt";
    } else {
        resultsFilePath = argv[3];
    }

    char *matrixFilePath = argv[1];
    int numberOfThreads = atoi(argv[2]);
    
    FILE *file;
    int lineCount = 0;

    file = fopen(matrixFilePath, "r");
    if (file == NULL) {
        printf("Can't open file: %s\n", matrixFilePath);
        return 1;
    }

    for (char c = getc(file); c != EOF; c = getc(file)) {
        if (c == '\n') {
            lineCount++;
        }
    }

    fclose(file);

    int numberOfMatrices = 0;
    int rows, cols;
    double **matrix, **matrixA, **matrixB;
    int matrixARows, matrixBRows, matrixACols, matrixBCols;
    file = fopen(matrixFilePath, "r");
    for (int i = 0; i < lineCount;) {
        if (fscanf(file, "%d, %d", &rows, &cols) == -1) {
            i++; // We're on an empty line skip the row
        } else {
            numberOfMatrices++;
            matrix = malloc(rows * sizeof(double*)); // Allocate rows
            for (int row = 0; row < rows; row++) {
                matrix[row] = malloc(cols * sizeof(double));// Alocate cols
                for (int col = 0; col < cols - 1; col++) {
                    fscanf(file, "%lf,", &matrix[row][col]);
                }
                fscanf(file, "%lf", &matrix[row][cols - 1]);
                i++; // Go to the next line after scanning row
            }
            
            if (numberOfMatrices == 1) {
                matrixA = malloc(rows * cols * sizeof(double));
                memcpy(matrixA, matrix, sizeof(double) * rows * cols);
                
                matrixARows = rows;
                matrixACols = cols;
                
                free(matrix);
            } else if (numberOfMatrices == 2) {
                matrixB = malloc(rows * cols * sizeof(double));
                memcpy(matrixB, matrix, sizeof(double) * rows * cols);

                matrixBRows = rows;
                matrixBCols = cols;
                numberOfMatrices = 0;
                
                free(matrix);
                
                if (canMultiplyMatrices(matrixACols, matrixBRows)) {
                    int computationCount = matrixARows;
                    int tempNumberOfThreads = numberOfThreads;

                    if (numberOfThreads > computationCount) {
                        tempNumberOfThreads = computationCount;
                    }

                    int slices[tempNumberOfThreads];
                    int remainder = computationCount % tempNumberOfThreads;

                    for (int i = 0; i < tempNumberOfThreads; i++) {
                        slices[i] = computationCount / tempNumberOfThreads;
                    }

                    for (int i = 0; i < remainder; i++) {
                        slices[i] = slices[i] + 1;
                    }

                    int startPoints[tempNumberOfThreads];
                    int endPoints[tempNumberOfThreads];

                    for (int i = 0; i < tempNumberOfThreads; i++) {
                        if (i == 0) {
                            startPoints[i] = 0;
                            endPoints[i] = startPoints[i] + slices[i] - 1;
                        } else {
                            startPoints[i] = endPoints[i - 1] + 1;
                            endPoints[i] = startPoints[i] + slices[i] - 1;
                        }
                    }

                    ThreadData data[tempNumberOfThreads];
                    pthread_t threads[tempNumberOfThreads];

                    double **resultMatrix = (double**)malloc(matrixARows * sizeof(double*));
                    for (i = 0; i < matrixARows; i++) {
                        resultMatrix[i] = (double*)malloc(matrixBCols * sizeof(double));
                    }

                    for (int i = 0; i < tempNumberOfThreads; i++) {
                        data[i].start = startPoints[i];
                        data[i].end = endPoints[i];
                        data[i].matrixA = matrixA;
                        data[i].matrixB = matrixB;
                        data[i].colsA = matrixACols;
                        data[i].colsB = matrixBCols;
                        data[i].rowsA = matrixARows;
                        data[i].rowsB = matrixBRows;
                        data[i].resultMatrix = resultMatrix;

                        pthread_attr_t attr;
                        pthread_attr_init(&attr);
                        pthread_create(&threads[i], &attr, multiplyMatrices, &data[i]);
                    }

                    for (int i = 0; i < tempNumberOfThreads; i++) {
                        pthread_join(threads[i], NULL);
                    }

                    printMatrix(resultMatrix, matrixARows, matrixBCols);

                    FILE *resultsFile = fopen(resultsFilePath, "a");
                    if (resultsFile == NULL) {
                        printf("Error opening results file!\n");
                        return 1; // TODO This wrong
                    }

                    fprintf(resultsFile, "%d, %d\n", matrixACols, matrixBRows);

                    for (int row = 0; row < matrixARows; row++) {
                        for (int col = 0; col < matrixBCols - 1; col++) {
                            fprintf(resultsFile, "%lf, ", resultMatrix[row][col]);
                        }
                        fprintf(resultsFile, "%lf\n", resultMatrix[row][cols - 1]);
                    }

                    fprintf(resultsFile, "----------------\n");
                    fclose(resultsFile);
                    free(resultMatrix);
                } else {
                    printf("Can't multiple MatrixA with MatrixB the cols from MatrixA must be equal to the rows of MatrixB!\n");
                }

                free(matrixA);
                free(matrixB);
            }
        }
        i++; // Go to the next line
    }
    
    fclose(file);

    return 0;
}

void *multiplyMatrices(void *threadArgs) {
    ThreadData *threadData = (ThreadData *)threadArgs;
    int start = threadData->start, end = threadData->end;
    int rowsA = threadData->rowsA, colsA = threadData->colsA;
    int rowsB = threadData->rowsB, colsB = threadData->colsB;
    double **matrixA = threadData->matrixA, **matrixB = threadData->matrixB, **resultMatrix = threadData->resultMatrix;

    for (int i = start; i <= end; i++) {
        for (int colB = 0; colB < colsB; colB++) {
            for (int colA = 0; colA < colsA; colA++) {
                pthread_mutex_lock(&mutex);
                resultMatrix[i][colB] += matrixA[i][colA] * matrixB[colA][colB];
                pthread_mutex_unlock(&mutex);
            }
        }
    }
}

void printMatrix(double **matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols - 1; col++) {
            printf("%lf,", matrix[row][col]);
        }
        printf("%lf\n", matrix[row][cols - 1]);
    }
}

bool canMultiplyMatrices(int matrixACols, int matrixBRows) {
    return matrixACols == matrixBRows;
}
