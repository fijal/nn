
#include "nn.h"

float act_func(float v)
{
	if (v > 0.0f)
		return 1.0f;
	return 0.0f;
}

float calculate_value(struct neuron* n)
{
	float v = 0.0f;
	for (int i = 0; i < n->num_connections; i++)
		v += n->connections[i]->value * n->weights[i];
	n->value = act_func(v);
	n->counter = 0;
	return v;
}
