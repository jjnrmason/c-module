/*
Your program will decode a PNG file into an array and apply the box blur filter. Blurring an image 
reduces noise by taking the average RGB values around a specific pixel and setting it’s RGB to the 
mean values you’ve just calculated. This smoothens the colour across a matrix of pixels. For this 
assessment, you will use a 3x3 matrix. For example, if you have a 5x5 image such as the following (be 
aware that the coordinate values will depend on how you format your 2D array):
0,4 1,4 2,4 3,4 4,4
0,3 1,3 2,3 3,3 4,3
0,2 1,2 2,2 3,2 4,2
0,1 1,1 2,1 3,1 4,1
0,0 1,0 2,0 3,0 4,0

The shaded region above represents the pixel we want to blur, in this case, we are focusing on pixel 
1,2 (x,y) (Centre of the matrix). to apply the blur for this pixel, you would sum all the Red values from 
the surrounding coordinates including 1,2 (total of 9 R values) and find the average (divide by 9). This 
is now the new Red value for coordinate 1,2. You must then repeat this for Green and Blue values.

This must be repeated throughout the image. If you are working on a pixel which is not fully 
surrounded by pixels (8 pixels), you must take the average of however many neighbouring pixels 
there are. 

NOTE – this program should work with any amount of threads.

Reading in an image file into a single or 2D array (10 marks)
Applying Box filter on image (20 marks)
Using multithreading appropriately to apply Box filter (40 marks)
Using dynamic memory – malloc (10 marks)
Outputting the correct image with Box Blur applied (20 marks)

1) take in an image
2) convert to a 2D array (you could keep it as a 1D array, but I imagine that would be much more pain and suffering)
3) blurring magic (which will later utilise threading)
4) output final result

Compile with "cc task4.c lodepng.c"
*/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "lodepng.h"

typedef struct {
    int thread;
    int start;
    int end;
    unsigned int width;
    unsigned int height;
    unsigned char* image;
} tdata;

void *blurImage(void* targs) {
    pthread_exit(0);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Program run with the incorrect args.\n");
        printf("Arg1: Image filename\n");
        printf("Arg2: New image filename\n");
        printf("Arg3: Number of threads\n");
        return 0;
    }

    char* filename = argv[1];
    char* newFilename = argv[2];
    int numberOfThreads = atoi(argv[3]);

    unsigned char* image;
    unsigned int width, height, error;

    error = lodepng_decode32_file(&image, &width, &height, filename);

    if (error) {
        printf("Error: %s\n", lodepng_error_text(error));
        return 0;
    }

    tdata threadData[numberOfThreads];
    pthread_t threads[numberOfThreads];
    int start = 0, end = 0;

    // Print 1D array
    for (int i = 0; i < 64; i = i + 4) {
        printf("%d %d %d %d\n", image[i], image[1 + i], image[2 + i], image[3 + i]);
    }

    // Convert to 2D array
    unsigned char image2D[height][width * 4];
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width * 4; col = col + 4) {
            image2D[row][col] = image[(row * width * 4) + col];
            image2D[row][col + 1] = image[(row * width * 4) + col + 1];
            image2D[row][col + 2] = image[(row * width * 4) + col + 2];
            image2D[row][col + 3] = image[(row * width * 4) + col + 3];
        }
    }

    // Print 2D array
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width * 4; col++) {
            printf("%d ", image2D[row][col]);
            if ((col + 1) % 4 == 0) {
                printf("|");
            }
        }
        printf("\n");
    }

    // Convert back to 1D array and making everything blue
    int counter = 0;
    int imageArea = width * height;
    unsigned char image1D[imageArea];
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width * 4; col++) {
            // Try to make it entierly RED
            if (col == 0) {
                printf("Col = %d\n", col);
            }
            printf("%d\n", col);
            image2D[row][col] = 255;
            image1D[counter] = image2D[row][col];
            counter++;
        }
    }

    // Print 1D array
    for (int i = 0; i < 64; i = i + 4) {
        printf("%d %d %d %d\n", image1D[i], image1D[1 + i], image1D[2 + i], image1D[3 + i]);
    }

    // for (int i = 0; i < numberOfThreads; i++) {
    //     if (i == 0) {
    //         start = 0
    //     } else {
    //         start += chunkSize;
    //     }

    //     end = start + chunkSize - 1;

    //     if (i == numberOfThreads - 1) {
    //         end = something hmm;
    //     }

    //     threadData[i].start = start;
    //     threadData[i].end = end;
    //     threadData[i].width = width;
    //     threadData[i].height = height;
    //     threadData[i].image = image;

    //     pthread_create(threads[i], NULL, blurImage, &threadData[i]);
    //     pthread_join(threads[i], NULL);
    // }

    lodepng_encode32_file(newFilename, image1D, width, height);

    free(image); // Free pointer
    return 0;
}