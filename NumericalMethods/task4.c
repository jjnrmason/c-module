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

Compile with "cc task4.c lodepng.c -lm"
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
} imagePixel;

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
    int start = 0, end = 0, counter = 0;
    int nPixels = width * height;

    // Convert to 1D array
    printf("1D Array\n");
    imagePixel image1D[nPixels];
    for (int i = 0; i < nPixels * 4; i = i + 4) {
        imagePixel img;

        img.r = image[i];
        img.g = image[1 + i];
        img.b = image[2 + i];
        img.a = image[3 + i];

        image1D[counter] = img;

        printf("%d %d %d %d\n", image1D[counter].r, image1D[counter].g, image1D[counter].b, image1D[counter].a);
        counter = counter + 1;
    }

    printf("Width = %d, Height = %d\n", width, height);
    printf("Area = %d\n", nPixels);
    printf("2D Array\n");

    // Convert to 2D array
    counter = 0;
    imagePixel ** image2D = malloc(height * sizeof(imagePixel*));
    imagePixel ** image2DBlur = malloc(height * sizeof(imagePixel*));

    for (int i = 0; i < height; i++) {
        image2D[i] = malloc(width * sizeof(imagePixel));
        image2DBlur[i] = malloc(width * sizeof(imagePixel));
    }

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {            
            image2D[row][col] = image1D[counter];

            printf("Row: %d, Col: %d, RGBA: %d,%d,%d,%d\n", row, col, image2D[row][col].r, image2D[row][col].g, image2D[row][col].b, image2D[row][col].a);
            counter++;
        }
    }

    // iterate 2d arr
    //      YYY 
    //      YXY
    //      YYY     
    // get all 8 things around it
    // add colour together and divide by 8
    // that gives value for r g b
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

                if (col == 0 && row == 0) {
                        printf("Added bottom\n");
                } 

                // Bottom right
                if (rightAvailable) {
                    bottomRightRed = image2D[row + 1][col + 1].r;
                    bottomRightGreen = image2D[row + 1][col + 1].g;
                    bottomRightBlue = image2D[row + 1][col + 1].b;
                    bottomRightOpacity = image2D[row + 1][col + 1].a;
                    numberOfSurroundingPixels++;

                    if (col == 0 && row == 0) {
                        printf("Added bottom right\n");
                    } 
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

                if (col == 0 && row == 0) {
                    printf("Added mid right\n");
                    printf("GREEN: %d\n", image2D[row][col + 1].g);
                } 
                
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
                    printf("Surrounding green pixel: %d\n", surroundingGreenPixels[i]);
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
            
            printf("Number of surrounding pixels, %d\n", numberOfSurroundingPixels);
            

            image2DBlur[row][col].r = round(redPixelAverage);
            image2DBlur[row][col].g = round(greenPixelAverage);
            image2DBlur[row][col].b = round(bluePixelAverage);
            image2DBlur[row][col].a = round(opacityPixelAverage);
        }
    }

    // Convert back to 1D array and then back to flat array to save
    imagePixel* image1DSecondEdition = malloc(nPixels * sizeof(imagePixel));
    counter = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {            
            image1DSecondEdition[counter] = image2DBlur[row][col];
            counter++;
        }
    }

    unsigned char* newImage = malloc(nPixels * sizeof(int) * 4);
    counter = 0;
    for (int i = 0; i < nPixels * 4; i = i + 4) {
        newImage[i] = image1DSecondEdition[counter].r;
        newImage[i + 1] = image1DSecondEdition[counter].g;
        newImage[i + 2] = image1DSecondEdition[counter].b;
        newImage[i + 3] = image1DSecondEdition[counter].a;
        counter++;
    }

    printf("Print new 1D array\n");
    for (int i = 0; i < nPixels * 4; i = i + 4) {
        printf("%d, %d, %d, %d\n", newImage[i], newImage[i + 1], newImage[i + 2], newImage[i + 3]);
    }

    lodepng_encode32_file(newFilename, newImage, width, height);

    free(image); // Free pointers
    free(newImage);
    free(image1DSecondEdition);
    free(image2DBlur);
    free(image2D);
    return 0;
}