import { Matrix } from './Matrix';

// Declare map functions
function random(_x: any) {
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
    let layers = [this.inputNum, ...this.hiddenNums, this.outputNum];
    for (let l = 0; l < layers.length - 1; l++) {
      let w = new Matrix(layers[l + 1], layers[l]);
      w.map(random);
      this.weights.push(w);

      let b = new Matrix(layers[l + 1], 1);
      b.map(random);
      this.biases.push(b);
    }

    this.lr = lr;
  }

  feedForward(inputs: number[]) {
    // Convert inputs into a Matrix object
    const inputsMatrix = Matrix.fromArray(inputs, this.inputNum, 1);
    let m: Matrix = inputsMatrix;

    for (let l = 0; l < this.weights.length; l++) {
      m = Matrix.mult(this.weights[l], m);
      m.addMatrix(this.biases[l]);
      m.map(sigmoid);
    }
    
    return m.toArray().reduce((accumulator, value) => accumulator.concat(value), []);
  }
}
