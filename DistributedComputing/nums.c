#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void countNums(float* nums, int chunksize, float* numCount) {
    for (int i = 0; i < chunksize; i++) {
        float f = nums[i];
        *numCount += f;
    }
}

void findAverage(int chunksize, float total, float* average) {
    *average = total / chunksize;
}

void findMax(float* nums, int chunksize, float* maxNum) {
    for (int i = 0; i < chunksize; i++) {
        float f = nums[i];
        if (f > *maxNum) {
            *maxNum = f;
        }
    }
}

void findMin(float* nums, int chunksize, float* minNum) {
    for (int i = 0; i < chunksize; i++) {
        float f = nums[i];
        if (f < *minNum) {
            *minNum = f;
        }
    }
}

int main(int argc, char** argv) {
    FILE* file;
    int count = 0;
    int size;
    int rank;
    int chunksize;

    float val = 0.0;
    float* nums;
    char c;

    float numCount;
    float totalNumCount = 0.0;
    float numAverage;
    float finalNumAverage = 0.0;
    float maxNum;
    float finalMaxNum = 0.0;
    float minNum;
    float finalMinNum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    numCount = 0.0;
    numAverage = 0.0;
    maxNum = 0.0;
    minNum = 0.0;

    if (size < 2 || size > 10) {
        printf("You must use a minimum of 2 processes and maximum of 10.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        file = fopen("NumbersForMPI.txt", "r");

        if (NULL == file) {
            printf("File can't be opened!\n");
            MPI_Finalize();
            return 1;
        }

        printf("File opened\n");

        while (!feof(file)){
            if (fscanf(file, "%f%c", &val, &c) > 0) {
                count++;
            }
        }

        fclose(file);

        printf("File closed\n");
        printf("File contained %d numbers\n", count);

        printf("Reading file contents into memory\n");
        nums = malloc(count * sizeof(float));
        file = fopen("NumbersForMPI.txt", "r");

        for (int i = 0; i < count; i++) {
            fscanf(file, "%f%c", &(nums[i]), &c);
        }

        fclose(file);
        printf("File contents read into memory\n");

        printf("Total number of communicators = %d\n", size);
        chunksize = count / size;
        printf("Content chunksize = %d\n", chunksize);

        for (int i = 1; i < size; i++) {
            MPI_Send(&chunksize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&(nums[i * chunksize]), chunksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }

        countNums(nums, chunksize, &numCount);
        totalNumCount += numCount;

        findAverage(chunksize, numCount, &numAverage);
        finalNumAverage += numAverage;

        findMax(nums, chunksize, &maxNum);
        finalMaxNum = maxNum;

        findMin(nums, chunksize, &minNum);
        finalMinNum = minNum;

        for (int i = 1; i < size; i++) {
            MPI_Recv(&numCount, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            totalNumCount += numCount;

            MPI_Recv(&numAverage, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            finalNumAverage += numAverage;

            MPI_Recv(&maxNum, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (maxNum > finalMaxNum) {
                finalMaxNum = maxNum;
            }

            MPI_Recv(&minNum, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (minNum < finalMinNum) {
                finalMinNum = minNum;
            }
        }

        printf("Total count = %f\n", totalNumCount);

        finalNumAverage = finalNumAverage / size;
        printf("Final average = %f\n", finalNumAverage);

        printf("Final max num = %f\n", finalMaxNum);

        printf("Final min num = %f\n", finalMinNum);
    } else {
        MPI_Recv(&chunksize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nums = malloc(chunksize * sizeof(float));
        MPI_Recv(nums, chunksize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        countNums(nums, chunksize, &numCount);
        findAverage(chunksize, numCount, &numAverage);
        findMax(nums, chunksize, &maxNum);
        findMin(nums, chunksize, &minNum);

        MPI_Send(&numCount, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&numAverage, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&maxNum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&minNum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}