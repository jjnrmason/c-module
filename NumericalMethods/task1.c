/*
https://statisticsbyjim.com/regression/linear-regression-equation/
Compile with "cc task1.c"
Run with "./a.out"
*/
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    FILE* file;
    char* filename = "LinearRegressionDataFiles/datasetLR1.txt";
    float x, y;
    float sumx = 0, sumxsq = 0, sumy = 0, sumxy = 0;
    int coordinateCount = 0;

    file = fopen(filename, "r");

    if (file == NULL) {
        printf("Can't open file: %s\n", filename);
        return 0;
    }

    // Calculate all the sums of Σx, Σx2, Σy and Σxy to calculate a and b
    while (fscanf(file, "%f, %f", &x, &y) != EOF) {
        sumx = sumx + x; // Σx
        sumxsq = sumxsq + (x * x); // Σx2
        sumy = sumy + y; // Σy
        sumxy = sumxy + (x * y); // Σxy
        coordinateCount++;
    }

    fclose(file);

    printf("Sumx: %f\n", sumx);
    printf("Sumxsq: %f\n", sumxsq);
    printf("Sumy: %f\n", sumy);
    printf("Sumxy: %f\n", sumxy);
    printf("N coordinates: %d\n", coordinateCount);

    // Complete equation after finding sums
    float divisionSum = coordinateCount * sumxsq - sumx * sumx; 
    float a = (sumy * sumxsq - sumx * sumxy) / divisionSum;
    float b = (coordinateCount * sumxy - sumx * sumy) / divisionSum;

    // Insert into the equation: y = a + bx
    printf("y = %f + %fx\n", a, b);

    float input;
    printf("Enter a number to represent the value of x: ");
-   scanf("%e", &input);

    printf("x = %f\n", input);
    printf("y = %f + (%f * %f)\n", a, b, input);

    float answer = (b * input) + a;
    printf("y = %f\n", answer);

    return 0;
}
