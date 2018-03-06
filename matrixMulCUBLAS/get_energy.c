#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char* argv[]){
	FILE* gpu = fopen("gpu_power_parsed", "r");
	char* bf = (char*) malloc(sizeof(char) * 512);
	size_t x = 0;
	float gpu_energy = 0.0;
	while(getline(&bf, &x, gpu) != -1){
		int i = 0;
		gpu_energy += 0.1*atof(&bf[i]);
	}
	fclose(gpu);
	printf("GPU energy: %.2f.\n", gpu_energy);

}
