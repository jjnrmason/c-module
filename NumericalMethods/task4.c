/*
Compile with "cc task4.c lodepng.c -lm"
Run with "./a.out Images/testpepe.png Images/result.png 5"
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
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

typedef struct {
    int r;
    int b;
    int g;
    int a;
} pixel;

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

    int nPixels = width * height;

    printf("Width = %d, Height = %d\n", width, height);
    printf("Area = %d\n", nPixels);
    
    // Setup 2D array with malloc
    pixel ** image2D = malloc(height * sizeof(pixel*));
    pixel ** image2DBlur = malloc(height * sizeof(pixel*));

    for (int i = 0; i < height; i++) {
        image2D[i] = malloc(width * sizeof(pixel));
        image2DBlur[i] = malloc(width * sizeof(pixel));
    }

    // Convert to 2D array
    int counter = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            image2D[row][col].r = image[counter];
            image2D[row][col].g = image[counter + 1];
            image2D[row][col].b = image[counter + 2];
            image2D[row][col].a = image[counter + 3];
            counter += 4;
        }
    }

    /*
    Implement threading 🤯
    */
    tdata threadData[numberOfThreads];
    pthread_t threads[numberOfThreads];
    int start = 0, end = 0;
    int chunksize = (nPixels * 4) / numberOfThreads;

    printf("Chunksize: %d %d\n", chunksize, nPixels); 

    /*
    Iterate over the 2D array and find all the values surrounding a pixel.
    Add those values together and then divide by the amount of pixels to get the average.
    That becomes the value for the current pixel.
    */
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {         
            unsigned char topLeftRed = -1;
            unsigned char topLeftGreen = -1;
            unsigned char topLeftBlue = -1;
            unsigned char topLeftOpacity = -1;

            unsigned char topMiddleRed = -1;
            unsigned char topMiddleGreen = -1;
            unsigned char topMiddleBlue = -1;
            unsigned char topMiddleOpacity = -1;

            unsigned char topRightRed = -1;
            unsigned char topRightGreen = -1;
            unsigned char topRightBlue = -1;
            unsigned char topRightOpacity = -1;

            unsigned char middleLeftRed = -1;
            unsigned char middleLeftGreen = -1;
            unsigned char middleLeftBlue = -1;
            unsigned char middleLeftOpacity = -1;

            unsigned char middleRightRed = -1;
            unsigned char middleRightGreen = -1;
            unsigned char middleRightBlue = -1;
            unsigned char middleRightOpacity = -1;

            unsigned char bottomLeftRed = -1;
            unsigned char bottomLeftGreen = -1;
            unsigned char bottomLeftBlue = -1;
            unsigned char bottomLeftOpacity = -1;

            unsigned char bottomMiddleRed = -1;
            unsigned char bottomMiddleGreen = -1;
            unsigned char bottomMiddleBlue = -1;
            unsigned char bottomMiddleOpacity = -1;

            unsigned char bottomRightRed = -1;
            unsigned char bottomRightGreen = -1;
            unsigned char bottomRightBlue = -1;
            unsigned char bottomRightOpacity = -1;

            bool leftAvailable = false, rightAvailable = false, topAvailable = false, bottomAvailable = false;
            int numberOfSurroundingPixels = 0;

            if (row != 0) {
                topAvailable = true;
            }

            if (row != height - 1) {
                bottomAvailable = true;
            }

            if (col != 0) {
                leftAvailable = true;
            }

            if (col != width - 1) {
                rightAvailable = true;
            }
            
            // On the top row
            if (bottomAvailable) {
                // Bottom left
                if (leftAvailable) {
                    bottomLeftRed = image2D[row + 1][col - 1].r;
                    bottomLeftGreen = image2D[row + 1][col - 1].g;
                    bottomLeftBlue = image2D[row + 1][col - 1].b;
                    bottomLeftOpacity = image2D[row + 1][col - 1].a;
                    numberOfSurroundingPixels++;
                }

                // Bottom middle
                bottomMiddleRed = image2D[row + 1][col].r;
                bottomMiddleGreen = image2D[row + 1][col].g;
                bottomMiddleBlue = image2D[row + 1][col].b;
                bottomMiddleOpacity = image2D[row + 1][col].a;
                numberOfSurroundingPixels++;

                // Bottom right
                if (rightAvailable) {
                    bottomRightRed = image2D[row + 1][col + 1].r;
                    bottomRightGreen = image2D[row + 1][col + 1].g;
                    bottomRightBlue = image2D[row + 1][col + 1].b;
                    bottomRightOpacity = image2D[row + 1][col + 1].a;
                    numberOfSurroundingPixels++;
                }
            }

            // Middle left
            if (leftAvailable) {
                middleLeftRed = image2D[row][col - 1].r;
                middleLeftGreen = image2D[row][col - 1].g;
                middleLeftBlue = image2D[row][col - 1].b;
                middleLeftOpacity = image2D[row][col - 1].a;
                numberOfSurroundingPixels++;
            }

            // Middle right
            if (rightAvailable) {
                middleRightRed = image2D[row][col + 1].r;
                middleRightGreen = image2D[row][col + 1].g;
                middleRightBlue = image2D[row][col + 1].b;
                middleRightOpacity = image2D[row][col + 1].a;
                numberOfSurroundingPixels++;                
            }

            // On the bottom row
            if (topAvailable) {
                // Top left
                if (leftAvailable) {
                    topLeftRed = image2D[row - 1][col - 1].r;
                    topLeftGreen = image2D[row - 1][col - 1].g;
                    topLeftBlue = image2D[row - 1][col - 1].b;
                    topLeftOpacity = image2D[row - 1][col - 1].a;  
                    numberOfSurroundingPixels++;
                }

                // Top middle
                topMiddleRed = image2D[row - 1][col].r;
                topMiddleGreen = image2D[row - 1][col].g;
                topMiddleBlue = image2D[row - 1][col].b;
                topMiddleOpacity = image2D[row - 1][col].a;
                numberOfSurroundingPixels++;

                // Top right
                if (rightAvailable) {
                    topRightRed = image2D[row - 1][col + 1].r;
                    topRightGreen = image2D[row - 1][col + 1].g;
                    topRightBlue = image2D[row - 1][col + 1].b;
                    topRightOpacity = image2D[row - 1][col + 1].a;
                    numberOfSurroundingPixels++;
                }
            }

            unsigned char surroundingRedPixels[] = {
                topLeftRed,
                topMiddleRed,
                topRightRed,
                middleLeftRed,
                middleRightRed,
                bottomLeftRed,
                bottomMiddleRed,
                bottomRightRed
            };

            unsigned char surroundingGreenPixels[] = {
                topLeftGreen,
                topMiddleGreen,
                topRightGreen,
                middleLeftGreen,
                middleRightGreen,
                bottomLeftGreen,
                bottomMiddleGreen,
                bottomRightGreen
            };

            unsigned char surroundingBluePixels[] = {
                topLeftBlue,
                topMiddleBlue,
                topRightBlue,
                middleLeftBlue,
                middleRightBlue,
                bottomLeftBlue,
                bottomMiddleBlue,
                bottomRightBlue
            };

            unsigned char surroundingOpacityPixels[] = {
                topLeftOpacity,
                topMiddleOpacity,
                topRightOpacity,
                middleLeftOpacity,
                middleRightOpacity,
                bottomLeftOpacity,
                bottomMiddleOpacity,
                bottomRightOpacity
            };
            
            int redPixelTotal = 0;
            int greenPixelTotal = 0;
            int bluePixelTotal = 0;
            int opacityPixelTotal = 0;

            // Get total sum
            for (int i = 0; i < 8; i++) {
                if (surroundingRedPixels[i] != -1 && surroundingGreenPixels[i] != -1 && surroundingBluePixels[i] != -1 && surroundingOpacityPixels[i] != -1) {
                    redPixelTotal += surroundingRedPixels[i];
                    greenPixelTotal += surroundingGreenPixels[i];
                    bluePixelTotal += surroundingBluePixels[i];
                    opacityPixelTotal += surroundingOpacityPixels[i];
                }
            }

            // Get average
            float redPixelAverage = redPixelTotal / numberOfSurroundingPixels;
            float greenPixelAverage = greenPixelTotal / numberOfSurroundingPixels;
            float bluePixelAverage = bluePixelTotal / numberOfSurroundingPixels;
            float opacityPixelAverage = opacityPixelTotal / numberOfSurroundingPixels;
            
            image2DBlur[row][col].r = round(redPixelAverage);
            image2DBlur[row][col].g = round(greenPixelAverage);
            image2DBlur[row][col].b = round(bluePixelAverage);
            image2DBlur[row][col].a = round(opacityPixelAverage);
        }
    }

    // Convert back to flat array to save
    unsigned char* newImage = malloc(nPixels * sizeof(int) * 4);
    counter = 0;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            newImage[counter] = image2DBlur[row][col].r;
            newImage[counter + 1] = image2DBlur[row][col].g;
            newImage[counter + 2] = image2DBlur[row][col].b;
            newImage[counter + 3] = image2DBlur[row][col].a;
            counter += 4;
        }
    }

    // Output new image
    error = lodepng_encode32_file(newFilename, newImage, width, height);

    if (error) {
        printf("Error: %s\n", lodepng_error_text(error));
    }

    for (int i = 0; i < height; i++) {
    	free(image2D[i]);
    	free(image2DBlur[i]);
    }

    // Free pointers
    free(image); 
    free(newImage);
    free(image2D);
    free(image2DBlur);
    
    return 0;
}