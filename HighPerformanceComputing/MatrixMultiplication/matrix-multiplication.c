#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Program ran with the incorrect number of args.\n");
        printf("./a.out {{MatrixFilePath}} {{NumberOfThreads}}\n");
        return 1;
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
    printf("File contains %d lines.\n", lineCount);

    int numberOfMatrices = 0;
    int rows, cols;
    double **matrix, **matrixA, **matrixB; // TODO MatrixCompare MatrixToCompare?
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
                if (matrixACols == matrixBRows) {
                    double results[matrixARows][matrixBCols];
                    for (int rowA = 0; rowA < matrixARows; rowA++) {
                        for (int colB = 0; colB < matrixBCols; colB++) {
                            for (int colA = 0; colA < matrixACols; colA++) {
                                results[rowA][colB] += matrixA[rowA][colA] * matrixB[colA][colB];
                            }
                        }
                    }

                    FILE *resultsFile = fopen("results.txt", "a");
                    if (resultsFile == NULL) {
                        printf("Error opening results file!\n");
                        return 1; // This wrong
                    }

                    fprintf(resultsFile, "%d, %d * %d, %d\n", matrixARows, matrixACols, matrixBRows, matrixBCols);

                    for (int row = 0; row < matrixARows; row++) {
                        for (int col = 0; col < matrixBCols - 1; col++) {
                            printf("%lf,", results[row][col]);
                            fprintf(resultsFile, "%lf, ", results[row][col]);
                        }
                        printf("%lf\n", results[row][cols - 1]);
                        fprintf(resultsFile, "%lf\n", results[row][cols - 1]);
                    }

                    fprintf(resultsFile, "----------------\n");
                    fclose(resultsFile);
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