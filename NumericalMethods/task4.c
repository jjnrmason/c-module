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
    imagePixel image2D[height][width];
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {            
            image2D[row][col] = image1D[counter];

            printf("Row: %d, Col: %d, RGBA: %d,%d,%d,%d\n", row, col, image2D[row][col].r, image2D[row][col].g, image2D[row][col].b, image2D[row][col].a);
            counter++;
        }
    }

    printf("Print blue\n");
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {            
            image2D[row][col].r = 0;
            image2D[row][col].g = 0;
            image2D[row][col].b = 255;
            image2D[row][col].a = 255;

            printf("Row: %d, Col: %d, RGBA: %d,%d,%d,%d\n", row, col, image2D[row][col].r, image2D[row][col].g, image2D[row][col].b, image2D[row][col].a);
        }
    }
// iterate 2d arr
//      YYY 
//      YXY
//      YYY     
// get all 8 things around it
// add them together and divide by 8
// that gives value for r g b
// apart from the top and sides etc
    // Convert back to 1D array and then back to flat array to save
    imagePixel image1DSecondEdition[nPixels];
    counter = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {            
            image1DSecondEdition[counter] = image2D[row][col];
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
    return 0;
}