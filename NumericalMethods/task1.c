/*
https://statisticsbyjim.com/regression/linear-regression-equation/
*/
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    FILE* file;
    char* filename = "LinearRegressionDataFiles/datasetLR1.txt";
    float x, y, n;
    float sumx, sumxsq, sumy, sumxy;

    printf("Enter the value of 'n'(Number of points): ");
    scanf("%e", &n);
    printf("Calculating the linear regression with the value of 'n' being '%.0f'\n", n);

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
    }

    fclose(file);

    float divisionSum = n * sumxsq - sumx * sumx; 
    float a = (n * sumxy - sumx * sumy) / divisionSum;
    float b = (sumy * sumxsq - sumx * sumxy) / divisionSum;

    printf("a = %.0f, b = %.0f\n", a, b);
    // Insert into the equation: y = a + bx
    printf("y = %.0f + %.0fx\n", a, b);

    return 0;
}
