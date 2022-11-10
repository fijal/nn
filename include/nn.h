

typedef struct neuron {
	int num_connections;
	float *weights;
	struct neuron **connections;
	float value;
	float counter;
} neuron;

float calculate_value(struct neuron *);
