import { Matrix } from './Matrix';

// Declare map functions
function random(_x: number) {
  return Math.random();
}

function dsigmoid(x: number) {
  return x * (1 - x);
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

function relu(x: number) {
  return x > 0 ? x : 0;
}

function squared(x: number) {
  return x * x;
}

export class NeuralNetwork {
  inputNum: number;
  hiddenNums: number[];
  outputNum: number;
  weights: Matrix[] = [];
  biases: Matrix[] = [];
  lr: number;
  constructor(inNum: number, hidNums: number[], outNum: number, lr: number) {
    this.inputNum = inNum;
    this.hiddenNums = hidNums;
    this.outputNum = outNum;

    // Initialize weights and biases
    const layers = [this.inputNum, ...this.hiddenNums, this.outputNum];
    for (let l = 0; l < layers.length - 1; l++) {
      const w = new Matrix(layers[l + 1], layers[l]);
      w.map(random);
      this.weights.push(w);
      w.print();

      const b = new Matrix(layers[l + 1], 1);
      b.map(random);
      b.print();
      this.biases.push(b);
    }

    this.lr = lr;


  }

  feedForward(inputs: number[]) {
    // Convert inputs into a Matrix object
    const inputsMatrix = Matrix.fromArray(inputs, this.inputNum, 1);
    let m: Matrix = inputsMatrix;

    const layerActivations = [];

    for (let l = 0; l < this.weights.length; l++) {
      m = Matrix.mult(this.weights[l], m);
      m.addMatrix(this.biases[l]);
      m.map(sigmoid);

      layerActivations.push(m);
    }

    const outputs = m.toArray().reduce((accumulator, value) => accumulator.concat(value), []);

    return {
      outputs,
      layerActivations,
    }
  }

  train(inputs: number[], targets: number[]) {
    const errors = [];
    const outputs = this.feedForward(inputs);
    const outputsMatrix = Matrix.fromArray(outputs.outputs, this.outputNum, 1);
    const targetsMatrix = Matrix.fromArray(targets, this.outputNum, 1);

    // Calculate error
    const outputError = Matrix.subMatrix(targetsMatrix, outputsMatrix);
    errors.push(outputError);
    for (let i = 1; i < this.hiddenNums.length + 1; i++) {
      let error = Matrix.transpose(this.weights[this.weights.length - i]);
      error = Matrix.mult(error, errors[i - 1]);
      errors.push(error);
    }

    // Calculate gradients
    const gradients = [];
    for (let i = 0; i < this.hiddenNums.length + 1; i++) {
      const gradient = Matrix.map(outputs.layerActivations[outputs.layerActivations.length - i - 1], dsigmoid);
      gradient.hadamard(errors[i]);
      gradients.push(gradient);
    }

    // Calculate deltas
    const deltas = [];
    for (let i = 1; i < this.hiddenNums.length + 1; i++) {
      const delta = Matrix.mult(gradients[i], Matrix.transpose(outputs.layerActivations[i - 1]));
      delta.scale(this.lr);
      deltas.push(delta);
    }

    // Update weights and biases
    for (let i = 0; i < this.hiddenNums.length; i++) {
      this.weights[i].addMatrix(deltas[i]);
      this.biases[i].addMatrix(gradients[gradients.length - i - 1]);
    }

    let newErrors = [];
    for (let i = errors.length - 1; i >= 0; i--) {
      newErrors.push(errors[i].toArray());
    }

    return newErrors
  }
}

let nn = new NeuralNetwork(2, [2], 1, 1);

const trainingData = [
  {
    "inputs": [0, 0],
    "targets": [0],
  },
  {
    "inputs": [0, 1],
    "targets": [1],
  },
  {
    "inputs": [1, 0],
    "targets": [1],
  },
  {
    "inputs": [1, 1],
    "targets": [1],
  },
];

let i = 0;
while (true) {
  const data = trainingData[Math.floor(Math.random() * trainingData.length)];
  const errors = nn.train(data.inputs, data.targets);

  if (i % 100000 === 0) {
    console.log(`Inputs: ${data.inputs}`);
    console.log(`Outputs: ${nn.feedForward(data.inputs).outputs}`);
    console.log(`Errors: ${errors}`);
  }
  i++;
}