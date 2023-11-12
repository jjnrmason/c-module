#include <stdio.h>
#include <stdlib.h>

__device__
char* crypt(char *rawPassword) {
	static char newPassword[11];

	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for (int i = 0; i < 10; i++) {
		if (i >= 0 && i < 6) {
			if (newPassword[i] > 122) {
				newPassword[i] = (newPassword[i] - 122) + 97;
			} else if (newPassword[i] < 97) {
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		} else {
			if (newPassword[i] > 57) {
				newPassword[i] = (newPassword[i] - 57) + 48;
			} else if(newPassword[i] < 48) {
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}

	return newPassword;
}

__device__ 
int doStringsMatch(char* one, char* two, int length) {
	int result = 1;
	for (int i = 0; i < length; i++) {
		if (one[i] != two[i]) {
			result = 0;
			break;
		}
	}
	return result;
}

__global__ 
void decrypt(char *encryptedPassword, char *alphabet, char *numbers, char *decryptedPassword) {
	char plainPassword[4];
	plainPassword[0] = alphabet[blockIdx.x];
	plainPassword[1] = alphabet[blockIdx.y];
	plainPassword[2] = numbers[threadIdx.x];
	plainPassword[3] = numbers[threadIdx.y];

	char *potentialEncryptedPassword;
	potentialEncryptedPassword = crypt(plainPassword);

	if (doStringsMatch(encryptedPassword, potentialEncryptedPassword, 11) > 0) {
		for (int i = 0; i < 4; i++) {
			decryptedPassword[i] = plainPassword[i];
		}
	}
}

int main(int argc, char *argv[]) {
	char *cpuEncryptedPassword;
	if (argc < 2) {
		printf("Please provide your password to decrypt!\n");
		return 1;
	} else {
		cpuEncryptedPassword = argv[1];
	}

	char cpuAlphabet[26] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };
	char cpuNumbers[10] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

	int sizeOfEncryptedPassword = sizeof(char) * 11;
	int sizeOfAlphabet = sizeof(char) * 26;
	int sizeOfNumbers = sizeof(char) * 10;

	char *gpuEncryptedPassword;
	cudaMalloc((void **) &gpuEncryptedPassword, sizeOfEncryptedPassword);
	cudaMemcpy(gpuEncryptedPassword, cpuEncryptedPassword, sizeOfEncryptedPassword, cudaMemcpyHostToDevice);
	
	char *gpuAlphabet;
	cudaMalloc((void **) &gpuAlphabet, sizeOfAlphabet);
	cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeOfAlphabet, cudaMemcpyHostToDevice);

	char *gpuNumbers;
	cudaMalloc((void **) &gpuNumbers, sizeOfNumbers);
	cudaMemcpy(gpuNumbers, cpuNumbers, sizeOfNumbers, cudaMemcpyHostToDevice);

	char *gpuDecryptedPassword;
	cudaMalloc((void **) &gpuDecryptedPassword, sizeOfEncryptedPassword);

	decrypt<<<dim3(26, 26, 1), dim3(10, 10, 1)>>>(gpuEncryptedPassword, gpuAlphabet, gpuNumbers, gpuDecryptedPassword);
	cudaThreadSynchronize(); 

	printf("Finished synchronising threads\n");

	char *cpuDecryptedPassword = (char*)malloc(sizeof(char) * 4);
	cudaMemcpy(cpuDecryptedPassword, gpuDecryptedPassword, sizeOfEncryptedPassword, cudaMemcpyDeviceToHost);

	if (cpuDecryptedPassword != NULL || cpuDecryptedPassword[0] != 0) {
		printf("Found the password:\n");
		for (int i = 0; i < 4; i++) {
			printf("%s\n", cpuDecryptedPassword);
		}
	} else {
		printf("Couldn't find password!\n");
	}
	
	cudaFree(gpuEncryptedPassword);
	cudaFree(gpuAlphabet);
	cudaFree(gpuNumbers);
	cudaFree(gpuDecryptedPassword);
	free(cpuDecryptedPassword);
	return 0;
}
