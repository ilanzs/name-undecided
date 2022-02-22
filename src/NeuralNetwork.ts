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
  hiddenNum: number;
  outputNum: number;
  weightsIH: Matrix;
  weightsHO: Matrix;
  biasIH: Matrix;
  biasHO: any;
  lr: number;
  constructor(inNum: number, hidNum: number, outNum: number) {
    this.inputNum = inNum;
    this.hiddenNum = hidNum;
    this.outputNum = outNum;

    this.weightsIH = new Matrix(this.hiddenNum, this.inputNum);
    this.weightsIH.map(random);
    this.weightsHO = new Matrix(this.outputNum, this.hiddenNum);
    this.weightsHO.map(random);

    this.biasIH = new Matrix(this.hiddenNum, 1);
    this.biasIH.map(random);
    this.biasHO = new Matrix(this.outputNum, 1);
    this.biasHO.map(random);

    this.lr = 1;
  }

  feedForward(inputs: number[]) {
    // Convert inputs into a Matrix object
    const inputsMatrix = Matrix.fromArray(inputs, this.inputNum, 1);

    // Calculate the hidden layer output
    const hiddenInputs = Matrix.mult(this.weightsIH, inputsMatrix);
    const hiddenOutputs = Matrix.addMatrix(hiddenInputs, this.biasIH);
    const hiddenOutputsSigmoid = Matrix.map(hiddenOutputs, sigmoid);

    // Calculate the output layer output
    const outputInputs = Matrix.mult(this.weightsHO, hiddenOutputsSigmoid);
    const outputOutputs = Matrix.addMatrix(outputInputs, this.biasHO);
    const outputOutputsSigmoid = Matrix.map(outputOutputs, sigmoid);
    return outputOutputsSigmoid.toArray().reduce((accumulator, value) => accumulator.concat(value), []);
  }

  train(inputs: number[], targetsArr: number[]) {
    // Convert inputs into a Matrix object
    const inputsMatrix = Matrix.fromArray(inputs, this.inputNum, 1);

    // Calculate the hidden layer output
    const hiddenInputs = Matrix.mult(this.weightsIH, inputsMatrix);
    const hiddenOutputs = Matrix.addMatrix(hiddenInputs, this.biasIH);
    const hiddenOutputsSigmoid = Matrix.map(hiddenOutputs, sigmoid);

    // Calculate the output layer output
    const outputInputs = Matrix.mult(this.weightsHO, hiddenOutputsSigmoid);
    const outputOutputs = Matrix.addMatrix(outputInputs, this.biasHO);
    const outputOutputsSigmoid = Matrix.map(outputOutputs, sigmoid);

    // Convert arrays to Matrix objects
    const outputs = outputOutputsSigmoid;
    const targets = Matrix.fromArray(targetsArr, this.outputNum, 1);

    // Calculate the output errors
    const outputErrors = Matrix.subMatrix(targets, outputs);

    // Calculate the gradient of the output errors
    const gradients = Matrix.map(outputs, dsigmoid);
    gradients.hadamard(outputErrors);
    gradients.scale(this.lr);
    const hiddenTransposed = Matrix.transpose(hiddenOutputsSigmoid);
    const weightsHODeltas = Matrix.mult(gradients, hiddenTransposed);

    // Update the weights of the output layer
    this.weightsHO.addMatrix(weightsHODeltas);

    // Update the biases of the outpu layer
    this.biasHO.addMatrix(gradients);

    // Calculate the hidden errors
    const weightsHOTransposed = Matrix.transpose(this.weightsHO);
    const hiddenErrors = Matrix.mult(weightsHOTransposed, outputErrors);

    // Calculate the gradient of the hidden errors
    const hiddenGradients = Matrix.map(hiddenOutputsSigmoid, dsigmoid);
    hiddenGradients.hadamard(hiddenErrors);
    hiddenGradients.scale(this.lr);
    const inputsTransposed = Matrix.transpose(inputsMatrix);
    const weightsIHDeltas = Matrix.mult(hiddenGradients, inputsTransposed);

    // Update the weights of the hidden layer
    this.weightsIH.addMatrix(weightsIHDeltas);

    // Update the biases of the hidden layer
    this.biasIH.addMatrix(hiddenGradients);
  }
}
