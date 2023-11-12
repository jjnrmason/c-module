#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>

typedef struct {
    int start;
    int end;
    char *encryptedPasswordWithSalt;
} ThreadData;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int count = 0; // A counter used to track the number of combinations explored so far
bool found = false; // A global trigger set when the password has been found

/**
 Required by lack of standard function in C.   
*/
void substr(char *dest, char *src, int start, int length) {
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the 
 start of the line. Note that one of the most time consuming operations that 
 it performs is the output of intermediate results, so performance experiments 
 for this kind of program should not include this. i.e. comment out the printfs.
*/
void *crack(void *threadArgs) {
    ThreadData *data = (ThreadData *)threadArgs;
    int start = data->start;
    int end = data -> end;
    char *encryptedPasswordWithSalt = data -> encryptedPasswordWithSalt;

    char salt[7]; // String used in hashing the password. Need space for \0 // incase you have modified the salt value, then should modifiy the number accordingly
    char plain[7]; // The combination of letters currently being checked // Please modifiy the number when you enlarge the encrypted password.
    char *enc; // Pointer to the encrypted password

    substr(salt, encryptedPasswordWithSalt, 0, 6);

    printf("Start: %d and End: %d\n", (char)start, (char)end);

    for (int x = (char)start; x <= (char)end; x++) {
        if (found) {
            break;
        }
        for (int y = 'A'; y <= 'Z'; y++) {
            if (found) {
                break;
            }
            for (int z = 0; z <= 99; z++) {
                if (found) {
                    break;
                }

                sprintf(plain, "%c%c%02d", x, y, z); 
                pthread_mutex_lock(&mutex);
                enc = (char *) crypt(plain, salt);
                pthread_mutex_unlock(&mutex);
                count++;

                if (x == 'J' && y == 'T' && z == 13) {
                    printf("Plain: %s,\n Salt: %s\n enc: %s, \n Pass: %s\n", plain, salt, enc, encryptedPasswordWithSalt);
                }

                if (strcmp(encryptedPasswordWithSalt, enc) == 0) {
                    printf("#%-8d%s %s\n", count, plain, enc);
                    found = true;
                } else {
                    // printf("%-8d%s %s\n", count, plain, enc);
                }
            }
        }

        if (found) {
            pthread_cancel(pthread_self());
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Incorrect number of args passed!\n");
        printf("You should run with the args --numberOfThreads --encryptedPasswordWithSalt\n");
        printf("The password should be wrapped in single quotes\n");
        return 1;
    }

    int start, end;
    int numberOfThreads = atoi(argv[1]);
    char *encryptedPasswordWithSalt = argv[2];

    const int charCount = 26;
    int chunkSize = charCount / numberOfThreads;

    ThreadData data[numberOfThreads];
    pthread_t threads[numberOfThreads];

    for (int i = 0; i < numberOfThreads; i++) {
        // // Start at the beginning or the next chunk
        // if (i == 0) {
        //     start = 65;
        // } else {
        //     start += chunkSize;
        // }

        // end = start + chunkSize - 1;

        // // If we're on the last iteration
        // if (i == numberOfThreads - 1) {
        //     end = 90;
        // }

        start = ((i * chunkSize) + 65 + 1);
        end = ((i * chunkSize) + chunkSize + 65);

        if (i == 0) {
            start = 65;
        }

        if (i == numberOfThreads - 1) {
            end = 90;
        }

        data[i].start = start;
        data[i].end = end;
        data[i].encryptedPasswordWithSalt = encryptedPasswordWithSalt;
        pthread_create(&threads[i], NULL, crack, &data[i]);
    }

    for (int i = 0; i < numberOfThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("%d solutions explored\n", count);

    return 0;
}
