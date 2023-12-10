#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "lodepng.h"

typedef struct {
    int r;
    int g;
    int b;
    int a;
} Pixel;

__device__
Pixel* getFocusedValues() {
    Pixel* focusedValues = (Pixel*)malloc(sizeof(Pixel) * 9);
    
    for (int i = 0; i < 9; i++) {
        focusedValues[i].r = -1;
        focusedValues[i].g = -1;
        focusedValues[i].b = -1;
        focusedValues[i].a = -1;
    }

    return focusedValues;
}

__device__
Pixel getAverageOfPixels(Pixel* pixels, int numberOfPixels, int numberOfSurroundingPixels) {
    double totalRedPixels = 0.0, totalGreenPixels = 0.0, totalBluePixels = 0.0, totalOpacityPixels = 0.0;
    
    for (int i = 0; i < numberOfPixels; i++) {
        int r = pixels[i].r;
        int g = pixels[i].g;
        int b = pixels[i].b;
        int a = pixels[i].a;

        if (r <= 0 || g <= 0 || b <= 0 || a <= 0) {
            continue;
        }

        totalRedPixels += r;
        totalGreenPixels += g;
        totalBluePixels += b;
        totalOpacityPixels += a;
    }

    Pixel averageValues;
    
    averageValues.r = totalRedPixels / numberOfSurroundingPixels;
    averageValues.g = totalGreenPixels / numberOfSurroundingPixels;
    averageValues.b = totalBluePixels / numberOfSurroundingPixels;
    averageValues.a = totalOpacityPixels / numberOfSurroundingPixels;
    
    return averageValues;
}

__device__
Pixel getValuesAtIndex(unsigned char *originalImageValues, int index) {
    Pixel values;
    values.r = originalImageValues[index];
    values.g = originalImageValues[index + 1];
    values.b = originalImageValues[index + 2];
    values.a = originalImageValues[index + 3];
    return values;
}

__global__
void applyBlur(int height, int width, unsigned char *originalImageValues, unsigned char *blurredImageValues) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Pixel Coordinates
    int pixelX = threadIdx.x;
    int pixelY = blockIdx.x;
    int startingPixelIndex = id * 4;

    int topLeftStartingIndex = startingPixelIndex - width * 4 - (1 * 4); // Go back a row and then go left by one pixel
    int topMiddleStartingIndex = startingPixelIndex - width * 4; // Go back a row
    int topRightStartingIndex = startingPixelIndex - width * 4 + (1 * 4); // Go back a row and then go right by one pixel

    int middleLeftStartingIndex = startingPixelIndex - (1 * 4); // Go left by one
    int middleRightStartingIndex = startingPixelIndex + (1 * 4); // Go right by one

    int bottomLeftStartingIndex = startingPixelIndex + width * 4 - (1 * 4); // Go to next row and then go left by one pixel
    int bottomMiddleStartingIndex = startingPixelIndex + width * 4; // Go to next row
    int bottomRightStartingIndex = startingPixelIndex + width * 4 + (1 * 4); // Go to next row and then go right by one pixel

    // Store the pixel and surrounding values
    int numberOfSurroundingPixels = 0;
    Pixel* focusedValues = getFocusedValues();
    focusedValues[4] = getValuesAtIndex(originalImageValues, startingPixelIndex);
    numberOfSurroundingPixels++;

    bool leftAvailable = false, rightAvailable = false, topAvailable = false, bottomAvailable = false;

    if (pixelY != 0) {
        topAvailable = true;
    }

    if (pixelY != height - 1) {
        bottomAvailable = true;
    }

    if (pixelX != 0) {
        leftAvailable = true;
    }

    if (pixelX != width - 1) {
        rightAvailable = true;
    }

    // On the top row
    if (bottomAvailable) {
        // Bottom left
        if (leftAvailable) {
            focusedValues[6] = getValuesAtIndex(originalImageValues, bottomLeftStartingIndex);
            numberOfSurroundingPixels++;
        }

        // Bottom middle
        focusedValues[7] = getValuesAtIndex(originalImageValues, bottomMiddleStartingIndex);
        numberOfSurroundingPixels++;

        // Bottom right
        if (rightAvailable) {
            focusedValues[8] = getValuesAtIndex(originalImageValues, bottomRightStartingIndex);
            numberOfSurroundingPixels++;
        }
    }

    // Middle left
    if (leftAvailable) {
        focusedValues[3] = getValuesAtIndex(originalImageValues, middleLeftStartingIndex);
        numberOfSurroundingPixels++;
    }

    // Middle right
    if (rightAvailable) {
        focusedValues[5] = getValuesAtIndex(originalImageValues, middleRightStartingIndex);
        numberOfSurroundingPixels++;                
    }

    // On the bottom row
    if (topAvailable) {
        // Top left
        if (leftAvailable) {
            focusedValues[0] = getValuesAtIndex(originalImageValues, topLeftStartingIndex);
            numberOfSurroundingPixels++;
        }

        // Top middle
        focusedValues[1] = getValuesAtIndex(originalImageValues, topMiddleStartingIndex);
        numberOfSurroundingPixels++;

        // Top right
        if (rightAvailable) {
            focusedValues[2] = getValuesAtIndex(originalImageValues, topRightStartingIndex);
            numberOfSurroundingPixels++;
        }
    }

    Pixel averageValues = getAverageOfPixels(focusedValues, 9, numberOfSurroundingPixels);

    blurredImageValues[startingPixelIndex] = averageValues.r;
    blurredImageValues[startingPixelIndex + 1] = averageValues.g;
    blurredImageValues[startingPixelIndex + 2] = averageValues.b;
    blurredImageValues[startingPixelIndex + 3] = averageValues.a;

    free(focusedValues);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Program ran with the incorrect number of args.\n");
        printf("./a.out {{ImageFile}} {{OutputImageFile}}\n");
        return 1;
    }

    char *filename = argv[1];
    char *outputFilename = argv[2];

    unsigned char *cpuImageValues;
    unsigned int width, height, error;

    error = lodepng_decode32_file(&cpuImageValues, &width, &height, filename);

    if (error) {
        printf("Error: %s\n", lodepng_error_text(error));
        return 0;
    }

    const int totalCpuImagePixels = width * height * 4;
    
    unsigned char *gpuImageValues;
    cudaMalloc((void **)&gpuImageValues, sizeof(unsigned char) * totalCpuImagePixels);
    cudaMemcpy(gpuImageValues, cpuImageValues, sizeof(unsigned char) * totalCpuImagePixels, cudaMemcpyHostToDevice);
    
    unsigned char *gpuBlurredImageValues;
    cudaMalloc((void **)&gpuBlurredImageValues, sizeof(unsigned char) * totalCpuImagePixels);

    applyBlur<<<height, width>>>(height, width, gpuImageValues, gpuBlurredImageValues);
    cudaDeviceSynchronize();
    
    printf("Threads have synchronised and finished.\n");
    
    unsigned char *cpuBlurredImageValues = (unsigned char *)malloc(sizeof(unsigned char) * totalCpuImagePixels);
    cudaMemcpy(cpuBlurredImageValues, gpuBlurredImageValues, sizeof(unsigned char) * totalCpuImagePixels, cudaMemcpyDeviceToHost);

    error = lodepng_encode32_file(outputFilename, cpuBlurredImageValues, width, height);

    if (error) {
        printf("Error: %s\n", lodepng_error_text(error));
    }

    free(cpuBlurredImageValues);
    cudaFree(gpuImageValues);
    cudaFree(gpuBlurredImageValues);
}