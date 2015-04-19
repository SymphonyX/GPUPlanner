#include "State.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/******************************************************************
This file is the interface that's used to call functions from a DLL
*******************************************************************/


#define CALL __stdcall
#define EXPORT __declspec(dllexport)

using namespace std;

StateStruct *Goals;
StateStruct **texture, **Agents;
float* hMap;
int rows, columns, goalCount, maps;
vector<int> agentCount;

extern "C" void computeCostsCuda(StateStruct* texture, int rows, int columns, int locality, int agentsNumber, StateStruct* agents, int maxIterations);
extern "C" void printTexture(StateStruct*, int, int);
extern "C" void clearTextureValues(StateStruct* texture, int rows, int columns);
extern "C" void propagateUpdateAfterObstacleMovement(StateStruct *texture, int rows, int columns);
 

StateStruct initState(int x, int y, float g, float costToReach, bool inconsistent) {
	StateStruct state;
	state.x = x;
	state.y = y;
	state.g = g;
	state.costToReach = costToReach;
	state.predx = -1;
	state.predy = -1;
	state.inconsistent = inconsistent;
	return state;
}

/*Allocates memory to store each agents.
-int agents: the number of agents used for a given goal
-int mapNumber: the goal these agents correspond to (1 map -> 1 goal)*/
extern "C" EXPORT void allocAgentsMem(int agents, int mapNumber)
{
	if (Agents == NULL) {
		Agents = (StateStruct**) malloc(sizeof(StateStruct*)*maps);
	}
	Agents[mapNumber] = (StateStruct*)malloc(agents*sizeof(StateStruct));
	
	agentCount.push_back(agents);
}

//Allocates memory to store the goal states
extern "C" EXPORT void allocGoalsMem(int goals)
{
	free(Goals);
	Goals = (StateStruct*)malloc(goals*sizeof(StateStruct));
	goalCount = goals;
}

/*Creates a goal state and sets it in its corresponding state space
- x,y: location in the grid
- cost: cost to transition to this states (transitions don't need to be uniform)
- mapNumber: the copy of the state space this goal corresponds to
Note: I'm assuming one goal per copy of the state space, the behavior for more than one goal is undefined*/
extern "C" EXPORT void insertGoal(int x, int y, float cost, int mapNumber) {
	StateStruct goalState = initState(x, y, 0.0f, cost, false);
	texture[mapNumber][y*columns+x] = goalState;
}

/*Creates an agent state and sets it in its corresponding state space and sets its position in the Agent array.
- x,y: location in the grid
- cost: cost to transition to this states (transitions don't need to be uniform)
- mapNumber: the copy of the state space this goal corresponds to
- agentNumber: the position in the agent array*/
extern "C" EXPORT void insertStart(int x, int y, float cost, int agentNumber, int mapNumber) {
	StateStruct startState = initState(x, y, -3.0f, cost, false);
	texture[mapNumber][y*columns+x] = startState;
	Agents[mapNumber][agentNumber] = startState;
}

/*Sets the corresponding values into a state at x,y in mapNumber*/
extern "C" EXPORT void insertValuesInMap(int x, int y, float g, float cost, bool inconsistent, int mapNumber) {
	texture[mapNumber][y*columns+x].g = g;
	texture[mapNumber][y*columns+x].costToReach = cost;
	texture[mapNumber][y*columns+x].inconsistent = inconsistent;
}

/*After goal movement, all values are invalidates. This function clears those values for the corresponding map*/
extern "C" EXPORT void updateAfterGoalMovementCuda(int goalx, int goaly, int goalcost, int mapNumber)
{
	clearTextureValues(texture[mapNumber], rows, columns);
}

/*Plan repair after obstacle movement*/
extern "C" EXPORT void updateAfterObstacleMovementGrid(int mapNumber)
{
	propagateUpdateAfterObstacleMovement(texture[mapNumber], rows, columns);
}

/*Generates the state spaces used for planning.
- _rows: number of rows in the grid.
- _columns: number of columns in the grid.
- _maps: number of copy of state spaces (this corresponds to the number of goals)*/
extern "C" EXPORT void generateTexture(int _rows, int _columns, int _maps) {
	for (int index = 0; index < maps; index++) 
	{
		free(texture[index]);
	}
	maps = _maps;
	rows = _rows;
	columns = _columns;
	texture = (StateStruct**) malloc(sizeof(StateStruct*)*maps);
	size_t textureSize = ((rows*columns)*sizeof(StateStruct));

	for (int index = 0; index < maps; index++) {
		texture[index] = (StateStruct*) malloc(textureSize);
	}

	int i, j, m;
	for(m = 0; m < maps; m++) {
		for(i = 0; i < rows; i++) {
			for(j = 0; j < columns; j++) {
				StateStruct newState(j, i, -1.0f, 1.0f, -1, -1, false);
				texture[m][i*columns+j] = newState;
			}
		}
	}
}

/*Return a flatten array of all gvalues for a state space copy given by mapNumber*/
extern "C" EXPORT void CALL returnGMap(float gvalues[], int mapNumber) {
	int index = 0;
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < columns; ++j) {
			gvalues[index] = texture[mapNumber][i*columns+j].g;
			index++;
		}
	}
}


/*Return a flatten array of all transition costs for a state space copy given by mapNumber*/
extern "C" EXPORT void CALL returnCostMap(float costvalues[], int mapNumber) {
	int index = 0;
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < columns; ++j) {
			costvalues[index] = texture[mapNumber][i*columns+j].costToReach;
			index++;
		}
	}
}

/*Finds optimal solution by using the minimum required number of iterations*/
extern "C" EXPORT void computeCostsMinIndex(int mapNumber) {
	computeCostsCuda(texture[mapNumber], rows, columns, 2, agentCount[mapNumber], Agents[mapNumber], 0);
}

/*Finds the first solution it encounters*/
extern "C" EXPORT void computeCostsSubOptimal(int mapNumber) {
	computeCostsCuda(texture[mapNumber], rows, columns, 0, agentCount[mapNumber], Agents[mapNumber], 0);
}

/*Finds optimal solution by exploring the entire state space*/
extern "C" EXPORT void computeCostsOptimal(int mapNumber) {
	computeCostsCuda(texture[mapNumber], rows, columns, 1, agentCount[mapNumber], Agents[mapNumber], 0);
}

/*Performs 'iterations' number of iterations of the plannner*/
extern "C" EXPORT void computeIterations(int iterations, int mapNumber) {
	computeCostsCuda(texture[mapNumber], rows, columns, 1, agentCount[mapNumber], Agents[mapNumber], iterations);
}
