#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// Unoptimised matrix functions
// To do: create mt_compute, mt_del, allow option for output

typedef struct 
{
	int rows;
	int cols;
	double *data;
}Matrix;

Matrix mt_initialise(int rows, int cols) 
{
	Matrix *m = malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
	m->data = malloc(sizeof(double) * rows * cols);
	for (unsigned i = 0; i < rows * cols; i++) 
  {
 		m->data[i] = 0;
	}
	return *m;
}

Matrix mt_from_arr(double* arr, int arr_length) {
  Matrix m = mt_initialise(arr_length, 1);
  for (int i = 0; i < arr_length; i++) {
    m.data[i] = arr[i];
  }
  return m;
}

void mt_print(Matrix m) 
{
	for (int i = 0; i < m.rows * m.cols; i++) 
  {
		printf("%f ", m.data[i]);
		if ((i+1) % m.cols == 0) 
    {
			printf("\n");
		}
	}
}

Matrix mt_copy(Matrix m) {
  Matrix *new = malloc(sizeof(Matrix));
  new->rows = m.rows;
  new->cols = m.cols;
	new->data = malloc(sizeof(double) * m.rows * m.cols);
	for (unsigned i = 0; i < m.rows * m.cols; i++) 
  {
 		new->data[i] = m.data[i];
	}
	return *new;
}

// Sets all values of m to pseudo-random values between 0 and 1, leaving default seed for repeatibility
Matrix mt_randomise(Matrix m) 
{
	for (int i = 0; i < m.rows * m.cols; i++) 
  {
		m.data[i] = (double)rand()/RAND_MAX;
	}
	return m;
}

void mt_scalar_add(Matrix m, double num) 
{
	for (int i = 0; i < m.rows * m.cols; i++) 
  {
		m.data[i] += num;
	}
}

Matrix mt_add(Matrix m1, Matrix m2) 
{
	assert(m1.cols == m2.cols && m1.rows == m2.rows);
	Matrix new = mt_initialise(m1.rows, m1.cols);

	for (int i = 0; i < m1.rows * m1.cols; i++) 
  {
		new.data[i] = m1.data[i] + m2.data[i];
	}
	return new;
}

void mt_scalar_subtract(Matrix m, double num) 
{
	for (int i = 0; i < m.rows * m.cols; i++) 
  {
		m.data[i] -= num;
	}
}

Matrix mt_subtract(Matrix m1, Matrix m2) 
{
	assert(m1.cols == m2.cols && m1.rows == m2.rows);
	Matrix new = mt_initialise(m1.rows, m1.cols);

	for (int i = 0; i < m1.rows * m1.cols; i++) 
  {
		new.data[i] = m1.data[i] - m2.data[i];
	}
	return new;
}

void mt_scalar_multiply(Matrix m, double num) 
{
	for (int i = 0; i < m.rows * m.cols; i++) 
  {
		m.data[i] *= num;
	}
}

Matrix mt_multiply(Matrix m1, Matrix m2) 
{
	assert(m1.cols == m2.rows);
	Matrix m = mt_initialise(m1.rows, m2.cols);
	for (int i = 0; i < m1.rows; i++) 
  {
    for (int j = 0; j < m2.cols; j++) 
    {
      double sum = 0;
      for (int k = 0; k < m1.cols; k++) 
      {
        sum += m1.data[i * m1.cols + k] * m2.data[k * m2.cols + j];
      }
      m.data[i * m.cols + j] = sum;
    }
  }
	return m;
}

Matrix mt_multiply_elementwise(Matrix m1, Matrix m2) 
{
	assert(m1.cols == m2.cols && m1.rows == m2.rows);
	Matrix m = mt_initialise(m1.rows, m1.cols);

	for (int i = 0; i < m1.rows * m1.cols; i++) 
  {
		m.data[i] = m1.data[i] * m2.data[i];
	}
	return m;
}

Matrix mt_transpose(Matrix m1) 
{
	Matrix m = mt_initialise(m1.cols, m1.rows);
	for (int i = 0; i < m.rows; i++) 
  {
		for (int j = 0; j < m.cols; j++) 
    {
			m.data[i * m.cols + j] = m1.data[j * m1.cols + i];
		}
	}
	return m;
}

typedef struct 
{
  Matrix weights;
  Matrix bias;

  Matrix inputs;
  Matrix outputs;

  Matrix errors;
  Matrix gradients;
  Matrix deltas;

}Layer;

typedef struct 
{
  double learning_rate;
  int layer_num;
  Layer *layers;
}NeuralNetwork;

Layer initialise_layer(int output_size, int input_size) 
{
  Matrix weights = mt_initialise(output_size, input_size);
  Matrix bias = mt_initialise(output_size, 1);
  Layer l = {mt_randomise(weights), mt_randomise(bias)};
	return l;
}

NeuralNetwork initialise_network(int *layer_sizes, int layer_num, double learning_rate) 
{
  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  nn->learning_rate = learning_rate;
  nn->layer_num = layer_num;
  nn->layers = malloc(sizeof(Layer) * layer_num);
  for (int i = 1; i < layer_num; i++) 
  {
    Layer layer = initialise_layer(layer_sizes[i], layer_sizes[i-1]);
    nn->layers[i-1] = layer;
  }
  return *nn;
}

void mt_sigmoid(Matrix m) 
{
  for (int i = 0; i < m.rows * m.cols; i++) 
  {
    m.data[i] =  1/(1+exp(-1 * m.data[i]));
  }
}

void mt_dsigmoid(Matrix m) 
{
  for (int i = 0; i < m.rows * m.cols; i++) 
  {
    m.data[i] =  m.data[i] * (1-m.data[i]);
  }
}

void mt_ReLU(Matrix m) 
{
  for (int i = 0; i < m.rows * m.cols; i++) 
  {
    m.data[i] =  (m.data[i] < 0) ? 0 : m.data[i];
  }
}

void mt_dReLU(Matrix m) 
{
  for (int i = 0; i < m.rows * m.cols; i++) 
  {
    m.data[i] =  (m.data[i] < 0) ? 0 : 1;
  }
}

// Generates output of nn given a piece of data
Matrix feed_forward(NeuralNetwork nn, Matrix inputs) 
{
  Matrix prev_outputs = inputs;
  for (int i = 0; i < nn.layer_num - 1; i++) 
  {
    nn.layers[i].outputs = mt_multiply(nn.layers[i].weights, prev_outputs);
    nn.layers[i].outputs = mt_add(nn.layers[i].outputs, nn.layers[i].bias);
    mt_sigmoid(nn.layers[i].outputs);
    prev_outputs = nn.layers[i].outputs;
  }
  return prev_outputs;
}

// Performs backprop on one piece of training data
// To do: Avoid creating new matrices wherever possible, deallocate unused memory, 
// add functionality for different loss and activation functions, add batch training, 
// split into separate functions for more flexibility
void train(NeuralNetwork nn, Matrix inputs, Matrix targets) 
{
  Matrix outputs = feed_forward(nn, inputs);

  for (int i = nn.layer_num - 2; i >= 0; i--) 
  {
    Layer layer = nn.layers[i];

    if (i == nn.layer_num - 2) 
    {
      // Insert loss function here
      layer.errors = mt_subtract(targets, outputs);
    } else 
    {
      layer.errors = mt_multiply(mt_transpose(nn.layers[i+1].weights), nn.layers[i+1].errors);
    }

    layer.gradients = layer.outputs;
    mt_dsigmoid(layer.gradients); 
    layer.gradients = mt_multiply_elementwise(layer.gradients, layer.errors);
    mt_scalar_multiply(layer.gradients, nn.learning_rate);

    Matrix prev_values = (i == 0) ? inputs : nn.layers[i-1].outputs;
    layer.deltas = mt_multiply(layer.gradients, mt_transpose(prev_values));

    layer.weights = mt_add(layer.weights, layer.deltas);
    layer.bias = mt_add(layer.bias, layer.gradients);

    nn.layers[i] = layer;
  }
}

int main() 
{
	printf("howzit\n");

  int layer_sizes[4] = {2, 4, 4, 1};
  NeuralNetwork nn = initialise_network(layer_sizes, 4, 0.1);

  // Initialising training data for XOR, I'm sure there is a better way to do this
  Matrix training_inputs[4] = {
    mt_from_arr((double[2]){1, 1}, 2), 
    mt_from_arr((double[2]){1, 0}, 2), 
    mt_from_arr((double[2]){0, 1}, 2), 
    mt_from_arr((double[2]){0, 0}, 2)
  };

  Matrix training_targets[4] = {
    mt_from_arr((double[1]){0}, 1), 
    mt_from_arr((double[1]){1}, 1),
    mt_from_arr((double[1]){1}, 1),
    mt_from_arr((double[1]){0}, 1)
  };

  for (int i = 0; i < 100000; i++) 
  {
    // Train on random piece of data
    int rand_idx =  (int)((double)rand()/RAND_MAX * 4);
    train(nn, training_inputs[rand_idx], training_targets[rand_idx]);
  }

  printf("Trained:\n");
  for (int i = 0; i < 4; i++) 
  {
    mt_print(feed_forward(nn, training_inputs[i]));
  }

  return 0;
} 
