
import random
from _nn_cffi import lib, ffi
import sys

seed = random.randrange(sys.maxsize)
#seed = 1141490426980523047
random.seed(seed)
print("Seed was:", seed)

keepalive = set()
#seed(321)

def fully_connect(con_from, con_to, skip_bias=False):
    connections = ffi.new("struct neuron*[%d]" % len(con_from))
    for i in range(len(con_from)):
        connections[i] = con_from[i]
    keepalive.add(connections)
    for i in range(len(con_to) - skip_bias):
        n = con_to[i]
        n.num_connections = len(con_from)
        x = ffi.new("float[%d]" % len(con_from))
        n.weights = x
        keepalive.add(x)
        n.connections = connections
        for k in range(len(con_from)):
            n.weights[k] = random.random() * 2 - 1

def set_input(inp, v):
    assert len(inp) == len(v)
    for i in range(len(v)):
        inp[i].value = v[i]

LEARNING = 0.001

def grad_descent(out, exp, LEARNING=LEARNING):
    assert len(out) == len(exp)
    misses = 0
    for i in range(len(out)):
        if out[i].value == exp[i]:
            continue
        misses += 1
        for k in range(out[i].num_connections):
            if out[i].value > exp[i]:
                if out[i].connections[k].value > 0:
                    out[i].weights[k] -= LEARNING
                    out[i].connections[k].counter -= LEARNING * out[i].weights[k]
            else:
                if out[i].connections[k].value > 0:
                    out[i].weights[k] += LEARNING
                    out[i].connections[k].counter += LEARNING * out[i].weights[k]
    return misses

def grad_descent_hidden(out, LEARNING=LEARNING):
    for i in range(len(out) - 1):
        if out[i].counter == 0:
            continue
        for k in range(out[i].num_connections):
            if out[i].connections[k].value > 0:
                out[i].weights[k] += out[i].counter
                if out[i].counter > 0:
                    out[i].connections[k].counter += 1
                else:
                    out[i].connections[k].counter -= 1

def get_weights(row):
    l = []
    for item in row:
        l.append([item.weights[k] for k in range(item.num_connections)])
    return l

input = [ffi.new('struct neuron*') for i in range(3)]
output = [ffi.new('struct neuron*') for i in range(2)]
hidden1 = [ffi.new('struct neuron*') for i in range(3)]
hidden1[2].value = 1
#hidden2 = 
fully_connect(input, hidden1, True)
fully_connect(hidden1, output)

examples = [([0, 0], [1, 0]), ([1, 0], [0, 1]), ([0, 1], [0, 1]), ([1, 1], [1, 0])]
misses = 0
for k in range(1000):
    #print("Iteration %d" % k)
    if k % 200 == 199:
        print("Misses: ", misses)
        #print(get_weights(row))
        misses = 0
    ex = random.choice(examples)
    set_input(input, ex[0] + [1])
    for n in hidden1:
        f = lib.calculate_value(n)
    #    print(f)
    for n in output:
        lib.calculate_value(n)
    misses += grad_descent(output, ex[1])
    grad_descent_hidden(hidden1)
    #print("O", get_weights(output))
    #print("H", get_weights(hidden1))