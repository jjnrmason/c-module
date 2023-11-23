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
Pixel getAverageOfPixels(Pixel* pixels, int numberOfPixels, int numberOfSurroundingPixels) {
    double totalRedPixels = 0.0, totalGreenPixels = 0.0, totalBluePixels = 0.0, totalOpacityPixels = 0.0;
    
    for (int i = 0; i < numberOfPixels; i++) {
        totalRedPixels += pixels[i].r;
        totalGreenPixels += pixels[i].g;
        totalBluePixels += pixels[i].b;
        totalOpacityPixels += pixels[i].a;
    }

    Pixel averageValues;
    
    averageValues.r = totalRedPixels / numberOfSurroundingPixels;
    averageValues.g = totalGreenPixels / numberOfSurroundingPixels;
    averageValues.b = totalBluePixels / numberOfSurroundingPixels;
    averageValues.a = 255;
    
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
void applyBlur(int width, int height, unsigned char *originalImageValues, unsigned char *blurredImageValues) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Pixel Coordinates
    int pixelX = id / width;
    int pixelY = id % width;

    if (pixelX > width || pixelY > height) {
        printf("Coordinate can't be outside the bounds of the image.\n");
        return;
    }

    int startingPixelIndex = id * 4;
    if (startingPixelIndex < 0 || startingPixelIndex >= width * height * 4) {
        printf("Starting pixel can't be outside the bounds of the image.\n");
    }

    int numberOfSurroundingPixels = 0;
    // Store the pixel and surrounding values
    Pixel* focusedValues = (Pixel*)malloc(sizeof(Pixel) * 9);
    focusedValues[4] = getValuesAtIndex(originalImageValues, startingPixelIndex);
    numberOfSurroundingPixels++;

    bool leftAvailable = false, rightAvailable = false, topAvailable = false, bottomAvailable = false;
    int colArrayIndex = pixelY * 4;

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
            int bottomLeftIndex = ((pixelX + 1) * width * 4) + (colArrayIndex - (4 * 1)); // Add comment explaining this
            focusedValues[6] = getValuesAtIndex(originalImageValues, bottomLeftIndex);
            numberOfSurroundingPixels++;
        }

        // Bottom middle
        int bottomMiddleIndex = ((pixelX + 1) * width * 4) + colArrayIndex;
        focusedValues[7] = getValuesAtIndex(originalImageValues, bottomMiddleIndex);
        numberOfSurroundingPixels++;

        // Bottom right
        if (rightAvailable) {
            int bottomRightIndex = ((pixelX + 1) * width * 4) + (colArrayIndex + (4 * 1));
            focusedValues[8] = getValuesAtIndex(originalImageValues, bottomRightIndex);
            numberOfSurroundingPixels++;
        }
    }

    // Middle left
    if (leftAvailable) {
        int middleLeftIndex = (pixelX * width * 4) + (colArrayIndex - (4 * 1));
        focusedValues[3] = getValuesAtIndex(originalImageValues, middleLeftIndex);
        numberOfSurroundingPixels++;
    }

    // Middle right
    if (rightAvailable) {
        int middleRightIndex = (pixelX * width * 4) + (colArrayIndex + (4 * 1));
        focusedValues[5] = getValuesAtIndex(originalImageValues, middleRightIndex);
        numberOfSurroundingPixels++;                
    }

    // On the bottom row
    if (topAvailable) {
        // Top left
        if (leftAvailable) {
            int topLeftIndex = ((pixelX - 1) * width * 4) + (colArrayIndex - (4 * 1));
            focusedValues[0] = getValuesAtIndex(originalImageValues, topLeftIndex);
            numberOfSurroundingPixels++;
        }

        // Top middle
        int topMiddleIndex = ((pixelX - 1) * width * 4) + colArrayIndex;
        focusedValues[1] = getValuesAtIndex(originalImageValues, topMiddleIndex);
        numberOfSurroundingPixels++;

        // Top right
        if (rightAvailable) {
            int topRightIndex = ((pixelX - 1) * width * 4) + (colArrayIndex + (4 * 1));
            focusedValues[2] = getValuesAtIndex(originalImageValues, topRightIndex);
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

    applyBlur<<<dim3(width, 1, 1), dim3(height, 1, 1)>>>(width, height, gpuImageValues, gpuBlurredImageValues);
    cudaThreadSynchronize();

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