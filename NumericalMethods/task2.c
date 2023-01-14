#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>

// Sum used across threads
double sum = 0.0;

// Struct used to pass data to threads
typedef struct {
    int start;
    int end;
} threadData;

// Calculate the latest sum for the thread and add onto the total sum
void *calcPi(void *threadArgs) {
    // Cast args to struct to get details and which iteration where on in the thread
    threadData *data = (threadData *)threadArgs;
    int start = data->start;
    int end = data->end;

    for (int i = start; i < end; i++) {
        double term = pow(-1, i) / (2 * i + 1);
        sum += term;
        printf("Sum = %f\n", sum);
    }
    pthread_exit(0);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Program not run with the correct args.\n");
        printf("argv[1] = number of iterations\n");
        printf("argv[2] number of threads to use\n");
        return 0;
    }

    int iterations = atoi(argv[1]);
    int threadCount = atoi(argv[2]);

    printf("Running Leibniz Formula wth %d iterations ", iterations);
    printf("and using %d threads\n", threadCount);

    int start = 0, end = 0;
    int chunkSize = iterations / threadCount;

    threadData data[threadCount];
    pthread_t threads[threadCount];
    
    // Workout the start and end values for each thread and create them
    for (int i = 0; i < threadCount; i ++) {
        // Start at the begining or the next chunk
        if (i == 0) {
            start = 0;
        } else {
            start += chunkSize;
        }
        
        end = start + chunkSize - 1;

        // If we're on the last iteration
        if (i == threadCount - 1) {
            end = iterations;
        }

        data[i].start = start;
        data[i].end = end;
        pthread_create(&threads[i], NULL, calcPi, &data[i]);
    }
    
    for (int i = 0; i < threadCount; i++) {
        pthread_join(threads[i], NULL);
    }

    // Perform the last step of the formula
    double approxPi = 4 * sum;
    printf("Approx PI after %d iterations = %f\n", iterations, approxPi);

    return 0;
}