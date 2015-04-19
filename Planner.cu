#include <math.h>
#include "State.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
 
#define	STARTING_VALUE -1
#define OBSTACLE_VALUE -2
#define GOAL_VALUE -3
#define BLOCK_SIZE 256

__device__ void retrieveNeighborsLocation(StateStruct *state, Point *neighbors) {
	Point N;
	N.x = state->x; N.y = state->y-1;	*neighbors = N;
	Point S;
	S.x = state->x; S.y = state->y+1;	*(neighbors+1) = S;
	Point E;
	E.x = state->x-1; E.y = state->y;	*(neighbors+2) = E;
	Point W;
	W.x = state->x+1; W.y = state->y;	*(neighbors+3) = W;
	Point NE;
	NE.x = state->x-1; NE.y = state->y-1;	*(neighbors+4) = NE;
	Point NW;
	NW.x = state->x+1; NW.y = state->y-1;	*(neighbors+5) = NW;
	Point SE;
	SE.x = state->x-1; SE.y = state->y+1;	*(neighbors+6) = SE;
	Point SW; 
	SW.x = state->x+1; SW.y = state->y+1;	*(neighbors+7) = SW;
}

__device__ float distance(StateStruct *from, StateStruct *to) {
	if(from->x == to->x) {
		return fabs((float)(to->y - from->y));
	}
	else if(from->y == to->y) {
		return fabs((float)(to->x - from->x));
	}
	else {
		return sqrt(fabs((float)(to->x - from->x))*fabs((float)(to->x - from->x)) +
			fabs((float)(to->y - from->y))*fabs((float)(to->y - from->y)));
	}
}

__device__ StateStruct *retrieveStateAtLocation(Point location, StateStruct *texture, int rows, int columns) {
	StateStruct* ptr = &texture[location.x+columns*location.y];
	return ptr;
}

__device__ int withinBounds(Point pt, int rows, int columns) { 
	return ((pt.x >= 0 && pt.x < columns) && (pt.y >= 0 && pt.y < rows));
}

__device__ int stateNeedsUpdate(StateStruct* state) {
	return state->g == STARTING_VALUE || state->g == GOAL_VALUE;
}

__device__ int stateIsObstacle(StateStruct* state) {
	return state->costToReach > 10.0f;
}

__device__ int isGoalState(StateStruct* state) {
	return state->g == 0.0f;
}

//GPU kernel that computes the planner solution.
__global__ void computeCostsKernel(StateStruct *current_texture, StateStruct *texture_copy, int rows, int columns, int *check, int *locality, float maxCost, bool allAgentsReached) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < columns && y < rows) {
		Point pt; pt.x = x; pt.y = y;
		StateStruct *state = retrieveStateAtLocation(pt, current_texture, rows, columns);

		if (withinBounds(pt, rows, columns)) {
			if(!stateIsObstacle(state) && !isGoalState(state)) {
				//if the state is an obstacle, do not compute neighbors
				Point neighbors[8];
				retrieveNeighborsLocation(state, &neighbors[0]);

				int i;
				for (i = 0; i < 8; ++i) {
					if (withinBounds(neighbors[i], rows, columns)) {
						StateStruct *neighbor = retrieveStateAtLocation(neighbors[i], texture_copy, rows, columns);
						if (stateIsObstacle(neighbor)) //if neighbor is an obstacle, do not use it as a possible neighbor
							continue;
						float newg = neighbor->g + distance(neighbor, state) * state->costToReach;
						if ((newg < state->g || stateNeedsUpdate(state)) && !stateNeedsUpdate(neighbor)) {
							state->predx = neighbors[i].x;
							state->predy = neighbors[i].y;
							state->g = newg;
							if (*locality == 1) {
								*check = 0;
							} else if (*locality == 2) {
								if (state->g < maxCost || !allAgentsReached) {
									*check = 0;
								}
							} else if (*locality == 0 && allAgentsReached) {
								*check = 1;
							}
						}
					}
				}
				Point predPt; predPt.x = state->predx; predPt.y = state->predy;
				StateStruct *selectedPredecessorCopy = retrieveStateAtLocation(predPt, texture_copy, rows, columns);
				state->inconsistent = false;
				if ((selectedPredecessorCopy != NULL && selectedPredecessorCopy->inconsistent) || stateIsObstacle(selectedPredecessorCopy)) {
					//if predecessor from read-only is inconsistent - clear inconsistent flag in write-only and mark state as inconsistent in write-only
					current_texture[state->predy*columns+state->predx].inconsistent = false;
					state->inconsistent = true;
					state->g = STARTING_VALUE;
				} 
			}
		}
	}	
}

__global__ void checkForInconsistency(StateStruct* texture, int rows, int columns, int* flag) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < columns && y < rows) {
		StateStruct* state = &texture[y*columns+x];
		if (state->inconsistent) {
			*flag = 1;
		}
	}
}

double showMemoryUsage1()
{
	size_t free_byte;
	size_t total_byte;
	cudaMemGetInfo(&free_byte, &total_byte);

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f MB", used_db/1024.0);
	return used_db;
}

/*Finds agent largest g-value. This is used to stop the planner earlier and still get an optimal solution*/ 
float agentsMaxCost(StateStruct* texture, int columns, int agentCount, StateStruct* agents) {
	float maxCost = -10000.0f;
	for (int i = 0; i < agentCount; i++)  {
		StateStruct agent = texture[columns*agents[i].y+agents[i].x];
		if (agent.g > maxCost) {
			maxCost = agent.g;
		}
	}
	return maxCost;
}

bool agentsReached(StateStruct* texture, int columns, int agentCount, StateStruct* agents) {
	for (int i = 0; i < agentCount; i++) {
		StateStruct agent = texture[columns*agents[i].y+agents[i].x];
		if (agent.g < 0.0f) {
			return false;
		}
	}
	return true;
}

/*Method that calls the kernel. The blocks and grid sizes could be improved, depending on the kind of GPU you have.
Also, there are several things that can be done in the kernel to improve performance. We can also talk about it.*/
extern "C" int computeCostsCuda(StateStruct* texture, int rows, int columns, int locality, int agentCount, StateStruct* agents, int maxIterations = 0) {
	int *locality_dev, *consistencyCheck, *consistencyCheck_dev, *flag, *flag_dev;
	
	int blockLength = sqrt((double)BLOCK_SIZE); 
	int gridLength = ceil((double)rows/(double)blockLength);
	
	dim3 blocks(gridLength, gridLength, 1);
	dim3 threads(blockLength, blockLength, 1);

	
	StateStruct *texture_device, *texture_device_copy;
	cudaMalloc((void**)&texture_device, (rows*columns)*sizeof(StateStruct));
	cudaMalloc((void**)&texture_device_copy, (rows*columns)*sizeof(StateStruct));
	//make a two copies of the initial map
	cudaMemcpy(texture_device, texture, (rows*columns)*sizeof(StateStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(texture_device_copy, texture, (rows*columns)*sizeof(StateStruct), cudaMemcpyHostToDevice);


	cudaMalloc((void**)&locality_dev, sizeof(int));
	int* locality_ptr = (int*)malloc(sizeof(int));
	*locality_ptr = locality;

	cudaMalloc((void**)&consistencyCheck_dev, sizeof(int));
	consistencyCheck = (int*)malloc(sizeof(int));

	cudaMalloc((void**)&flag_dev, sizeof(int));
	flag = (int*)malloc(sizeof(int));

	int iterations = 0;
	do {
		//set flag to 0 to check for changes
		if (locality == 1 || locality == 2) {
			*consistencyCheck = 1;
		} else {
			*consistencyCheck = 0;
		}
		*flag = 0;
		cudaMemcpy(locality_dev, locality_ptr, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(consistencyCheck_dev, consistencyCheck, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(flag_dev, flag, sizeof(int), cudaMemcpyHostToDevice);

		bool allAgentsReached = agentsReached(texture, columns, agentCount, agents);
		float maxCost;
		if (allAgentsReached) {
			maxCost = agentsMaxCost(texture, columns, agentCount, agents);
		}
		computeCostsKernel<<<blocks, threads>>>(texture_device, texture_device_copy, rows, columns, consistencyCheck_dev, locality_dev, maxCost, allAgentsReached);
		
		checkForInconsistency<<<blocks, threads>>>(texture_device, rows, columns, flag_dev);
		
		StateStruct* temp = texture_device;
		texture_device = texture_device_copy;
		texture_device_copy = temp;
		iterations++;

		
		cudaMemcpy(texture, texture_device, (rows*columns)*sizeof(StateStruct), cudaMemcpyDeviceToHost);
		cudaMemcpy(consistencyCheck, consistencyCheck_dev, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(flag, flag_dev, sizeof(int), cudaMemcpyDeviceToHost);
	} while((*consistencyCheck == 0 || *flag == 1) && iterations != maxIterations);

	showMemoryUsage1();


	cudaFree(texture_device); cudaFree(texture_device_copy);

	printf("Result was: %i\n\n", *consistencyCheck);
	printf("Number of iterations: %i\n\n", iterations);

	return 1;
}

__global__ void clearTextureValuesKernel(StateStruct* texture, int rows, int columns) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < columns && y < rows) {
		StateStruct* state = &texture[y*columns+x];
		if (!stateIsObstacle(state)) {
			state->g = STARTING_VALUE;
			state->predx = state->predy = -1;
		}
	}

}

extern "C" void clearTextureValues(StateStruct* texture, int rows, int columns) {
	StateStruct* texture_dev;
	int blockLength = sqrt((double)BLOCK_SIZE); 
	int gridLength = ceil((double)rows/(double)blockLength);
	dim3 blocks(gridLength, gridLength, 1);
	dim3 threads(blockLength, blockLength, 1);

	cudaMalloc((void**)&texture_dev, ((rows*columns)*sizeof(StateStruct)));
	cudaMemcpy(texture_dev, texture, (columns*rows)*sizeof(StateStruct), cudaMemcpyHostToDevice);
	clearTextureValuesKernel<<<blocks, threads>>> (texture_dev, rows, columns);
	cudaMemcpy(texture, texture_dev, (columns*rows)*sizeof(StateStruct), cudaMemcpyDeviceToHost);

	cudaFree(texture_dev);
}

__device__ bool equals(float a, float b)
{
	if (fabs(a - b) < 0.0001) {
		return true;
	}
	return false;
}

/*Kernel for update after obstacle movement.*/
__global__ void propagateAfterObstacleMovementKernel(StateStruct* texture, StateStruct* texture_copy, int* propagateUpdate, int rows, int columns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < columns && y < rows) {
		StateStruct state = texture_copy[y*columns+x];
		if (state.predx > -1 && state.predy > -1)
		{
			StateStruct predecessor = texture_copy[state.predy*columns+state.predx];
			float transitionCost = distance(&state, &predecessor) * state.costToReach;
			if ((stateIsObstacle(&predecessor) || stateIsObstacle(&state) || !equals(state.g, (predecessor.g + transitionCost))) && !isGoalState(&state))
			{
				texture[y*columns+x].predx = -1;
				texture[y*columns+x].predy = -1;
				texture[y*columns+x].g = STARTING_VALUE;
				
				*propagateUpdate = 1;
			}
		}
	}
}

extern "C" void propagateUpdateAfterObstacleMovement(StateStruct* texture, int rows, int columns)
{
	StateStruct* texture_dev, *texture_dev_copy;
	int blockLength = sqrt((double)BLOCK_SIZE); 
	int gridLength = ceil((double)rows/(double)blockLength);
	dim3 blocks(gridLength, gridLength, 1);
	dim3 threads(blockLength, blockLength, 1);

	cudaMalloc((void**)&texture_dev, ((rows*columns)*sizeof(StateStruct)));
	cudaMemcpy(texture_dev, texture, (columns*rows)*sizeof(StateStruct), cudaMemcpyHostToDevice);
		
	cudaMalloc((void**)&texture_dev_copy, ((rows*columns)*sizeof(StateStruct)));
	cudaMemcpy(texture_dev_copy, texture, (columns*rows)*sizeof(StateStruct), cudaMemcpyHostToDevice);

	int* propagateUpdate = (int*)malloc(sizeof(int));
	
	int* propagateUpdate_dev;
	cudaMalloc((void**)&propagateUpdate_dev, sizeof(int));
	do {
		*propagateUpdate = 0;
		cudaMemcpy(propagateUpdate_dev, propagateUpdate, sizeof(int), cudaMemcpyHostToDevice);

		propagateAfterObstacleMovementKernel<<<blocks, threads>>>(texture_dev, texture_dev_copy, propagateUpdate_dev, rows, columns);
		cudaMemcpy(texture_dev_copy, texture_dev, (rows*columns)*sizeof(StateStruct), cudaMemcpyDeviceToDevice);

		cudaMemcpy(propagateUpdate, propagateUpdate_dev, sizeof(int), cudaMemcpyDeviceToHost);
	} while(*propagateUpdate == 1);

	cudaMemcpy(texture, texture_dev, (columns*rows)*sizeof(StateStruct), cudaMemcpyDeviceToHost);

	cudaFree(texture_dev); cudaFree(texture_dev_copy); cudaFree(propagateUpdate_dev);
}