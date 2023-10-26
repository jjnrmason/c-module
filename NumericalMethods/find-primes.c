/*
Compile with "cc task3.c -pthread"
Run with "./a.out {{number of threads}}"
*/
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int thread;
    int start;
    int end;
} tdata;

FILE* filePtr;
FILE* resultsFilePtr;
int* testData;
int* foundPrimeNumbers;
char* filename = "PrimeNumberDataFiles/PrimeData1.txt";
char* resultsFilename = "PrimeNumbersResults.txt";
int totalCount = 0;
int count = 0;

// Calculate whether a number of prime or not. Returns 0 if the number is a prime number
int isPrime(int n) {
    int isPrime = 0;
    if (n == 0 || n == 1) {
        return 1;
    }
    
    for (int y = 2; y <= n / 2; y++) {
        if (n % y == 0) {
            isPrime = 1;
            break;
        }
    }

    return isPrime;
}

// Worker function for the thread
void *doWork(void* targs) {
    tdata* data = (tdata *)targs;
    int thread = data->thread;
    int start = data->start;
    int end = data->end;
    count = 0;

    printf("Thread %d started.\n", thread);

    // Foreach item in the splice work out whether it's prime and if it is add to the count
    for (int i = start; i < end; i++) {
        int res = isPrime(testData[i]);

        if (!res) {
            printf("Thread %d: %d is prime\n", thread, testData[i]);
            count++;
            fprintf(resultsFilePtr, "%d\n", testData[i]);
        }
    }

    // Return the count of prime numbers in the thread back to the worker thread
    pthread_exit(&count);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Program should be ran with a single arg = the number of threads to use\n");
        return 0;
    }

    int numberOfThreads = atoi(argv[1]);

    // Load numbers from file using malloc
    filePtr = fopen(filename, "r");
    if (filePtr == NULL) {
        printf("Can't open file: %s\n", filename);
        return 0;
    }

    int lineCount = 0, num = 0;

    while(fscanf(filePtr, "%d", &num) != EOF) {
        lineCount++;
    }

    fclose(filePtr);
    
    testData = malloc(sizeof(int) * lineCount);

    filePtr = fopen(filename, "r");
    if (filePtr == NULL) {
        printf("Can't open file: %s\n", filename);
        return 0;
    }

    // Using the total count of lines read in the contents
    for (int i = 0; i < lineCount; i++) {
        fscanf(filePtr, "%d", &testData[i]);
    }

    fclose(filePtr);

    resultsFilePtr = fopen(resultsFilename, "w");
    if (resultsFilePtr == NULL) {
        printf("Can't open file: %s\n", resultsFilename);
        return 0;
    }

    // Setup threads and calculate primes
    int start = 0, end = 0;
    int chunkSize = lineCount / numberOfThreads;

    tdata data[numberOfThreads];
    pthread_t threads[numberOfThreads];

    for (int i = 0; i < numberOfThreads; i++) {
        if (i == 0) {
            start = 0;
        } else {
            start += chunkSize;
        }

        end = start + chunkSize - 1;

        if (i == numberOfThreads - 1) {
            end = lineCount;
        }

        data[i].thread = i + 1;
        data[i].start = start;
        data[i].end = end;
        pthread_create(&threads[i], NULL, doWork, &data[i]);

        void* returnValue;

        pthread_join(threads[i], &returnValue);

        // When the thread is finished get the value returned, cast to an int and add to the total count
        int count = *(int *)returnValue;
        printf("Thread %d found %d prime numbers\n", i + 1, count);
        totalCount += count;
    }

    printf("Total count of prime numbers in the file is %d\n", totalCount);

    // Output results to the file. Creating a string with the total amount of numbers found.
    char fileEnd[100] = "Prime numbers found: \0";
    char countStr[10];
    sprintf(countStr, "%d", totalCount);
    strcat(fileEnd, countStr);
    fputs(fileEnd, resultsFilePtr);

    // Free up allocated memory and close file pointer
    free(testData);
    fclose(resultsFilePtr);

    return 0;
}