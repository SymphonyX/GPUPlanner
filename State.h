#ifndef STATE_H
#define STATE_H

struct Point {
	int x;
	int y;
};

/*This was my first time working with GPU. 
Now I know that this is not the best way of representing structs to work on GPU.
Poor memory layout that hurts performance. We can talk about it if needed.*/
struct StateStruct{
	StateStruct(){};
	StateStruct(int _x, int _y, float _g, float _costToReach, int _predx, int _predy, bool _inconsistent) {
		x = _x;
		y = _y;
		g = _g;
		costToReach = _costToReach;
		inconsistent = _inconsistent;
		predx = _predx;
		predy = _predy;
	}

	int x;
	int y;
	float g;
	float costToReach;
	int predx;
	int predy;
	bool inconsistent;
}; 

#endif