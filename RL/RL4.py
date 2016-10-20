import numpy as np
from Helper.BrainModule import BrainModule


class OutputNeuron(object):
    """
    The output neurons of the neural network. Update based on reward.
    """
    def __init__(self, input_n, weights=None):
        if not weights == None:
            self.weights = weights
        else:
            self.weights = np.ones(input_n)*0.5
        self.last_input = None
        self.output = 0

    def reward(self,reward,last_fired):
        reward = reward * 0.05
        self.weights[last_fired] += reward
        if reward > 0:
            for i,weight in enumerate(self.weights):
                if i != last_fired:
                    self.weights[i] += (-0.8 * reward)
        self.weights = np.clip(self.weights, 0.001, 0.999)

    def propagate(self,spike_input):
        self.output = (self.weights*np.array(spike_input)).sum()

class RL(BrainModule):
    """
    Simplistic neural networked Reinforcement Learning
    """
    def __init__(self, run_id, topology, load_weights=None):
        """
        run_id : ID for DragonflyBrain run
        topology: List with two elements, first being number of inputs, second being number of output neurons
        load_weights: File to load pre-learnt weights from
        """
        BrainModule.__init__(self, run_id)
        np.random.seed(1)

        input_n,output_n = topology
        self.topology = (input_n, output_n)

        if load_weights:
            weights = self.load_numpy_array(load_weights)
            self.output_layer = [OutputNeuron(input_n, weights[i]) for i in xrange(output_n)]
        else:
            self.output_layer = [OutputNeuron(input_n) for i in xrange(output_n)]
        self.chosen_output = None
        self.first_step = True
        self.last_fired = None
        self.last_input_was_zero = False

    def step(self,spike_input,reward=0.0):
        """
        Process an RL step, and generate an action from spike_input
        """
        res = [0,0,0,0]
        spike_input=np.array(spike_input)
        if spike_input.sum() == 0.0:
            return res

        if self.last_input_was_zero:
            reward = 0.0


        self.backwards(reward)
        self.forwards(spike_input)

        out = np.zeros(len(self.output_layer))
        for i,n in enumerate(self.output_layer):
            out[i] = n.output

        output_sum = out.sum()
        if output_sum != 0.0:
            out = out/output_sum
        else:
            out = np.ones(len(out))/float(len(out))

        r = np.random.rand()
        sum_out = 0
        for i,o in enumerate(out):
            sum_out += o
            if sum_out > r:
                self.chosen_output = i
                break

        self.last_fired = np.array(spike_input).argmax()

        if not spike_input.any():
            self.last_input_was_zero = True
        else:
            self.last_input_was_zero = False

        res[self.chosen_output] = 1
        return res

    def forwards(self,spike_input):
        """
        Process input forward
        """
        for i in self.output_layer:
            i.propagate(spike_input)


    def backwards(self, reward):
        """
        Update weights based on reward
        """
        if self.first_step:
            self.first_step = False
            return
        if self.chosen_output >= 0:
            self.output_layer[self.chosen_output].reward(reward, self.last_fired)

    def output_weights(self):
        """
        Output neural link weights
        """
        w = []
        for o in self.output_layer:
            w.append(o.weights)
        return w

    def save_weights(self, directory=None, run_id_prefix=False, name="rl_weights.dat"):
        """
        Save weights to file
        """
        w = self.output_weights()
        self.save_numpy_array(w, directory, name)
