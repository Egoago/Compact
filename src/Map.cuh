#ifndef _MAP_
#define _MAP_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

	void map(int* b, const int* a, unsigned int size);

#endif