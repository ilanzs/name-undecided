export class Matrix {
  rows: number;
  cols: number;
  vals: number[][];
  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.vals = [];
    for (let i = 0; i < rows; i++) {
      this.vals[i] = [];
      for (let j = 0; j < cols; j++) {
        this.vals[i][j] = 0;
      }
    }
  }

  print() {
    // tslint:disable-next-line:no-console
    console.table(this.vals);
  }

  set(row: number, col: number, val: number) {
    this.vals[row][col] = val;
  }

  map(f: (arg0: number) => number) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] = f(this.vals[i][j]);
      }
    }
  }

  addMatrix(other: Matrix) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] += other.vals[i][j];
      }
    }
  }

  addScaler(num: number) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] += num;
      }
    }
  }

  subMatrix(other: Matrix) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] -= other.vals[i][j];
      }
    }
  }

  subScaler(num: number) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] -= num;
      }
    }
  }

  hadamard(other: Matrix) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] *= other.vals[i][j];
      }
    }
  }

  mult(other: Matrix) {
    // Matrix product
    if (this.cols !== other.rows) {
      throw new Error('Columns of A must match rows of B.');
    }
    const result = new Matrix(this.rows, other.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        // Dot product of values in col
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.vals[i][k] * other.vals[k][j];
        }
        result.vals[i][j] = sum;
      }
    }
    this.vals = result.vals;
    this.rows = result.rows;
    this.cols = result.cols;
  }

  scale(num: number) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.vals[i][j] *= num;
      }
    }
  }

  transpose() {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.vals[i][j] = this.vals[j][i];
      }
    }
    this.vals = result.vals;
    this.rows = result.rows;
    this.cols = result.cols;
  }

  toArray() {
    return this.vals;
  }

  static fromArray(arr: number[], rows: number, cols: number) {
    const result = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.vals[i][j] = arr[i * cols + j];
      }
    }
    return result;
  }

  static print(m: Matrix) {
    m.print();
  }

  static set(m: Matrix, row: number, col: number, val: number) {
    const result = new Matrix(m.rows, m.cols);
    result.set(row, col, val);
    return result;
  }

  static map(m: Matrix, f: (arg0: number) => number) {
    const result = new Matrix(m.rows, m.cols);
    result.addMatrix(m);
    result.map(f);
    return result;
  }

  static addMatrix(m1: Matrix, m2: Matrix) {
    const result = new Matrix(m1.rows, m1.cols);
    result.addMatrix(m1);
    result.addMatrix(m2);
    return result;
  }

  static addScaler(m: Matrix, num: number) {
    const result = new Matrix(m.rows, m.cols);
    result.addScaler(num);
    result.addMatrix(m);
    return result;
  }

  static subMatrix(m1: Matrix, m2: Matrix) {
    const result = new Matrix(m1.rows, m1.cols);
    result.addMatrix(m1);
    result.subMatrix(m2);
    return result;
  }

  static subScaler(m: Matrix, num: number) {
    const result = new Matrix(m.rows, m.cols);
    result.addMatrix(m);
    result.subScaler(num);
    return result;
  }

  static hadamard(m1: Matrix, m2: Matrix) {
    const result = new Matrix(m1.rows, m1.cols);
    result.addMatrix(m1);
    result.hadamard(m2);
    return result;
  }

  static mult(m1: Matrix, m2: Matrix) {
    const result = new Matrix(m1.rows, m1.cols);
    result.addMatrix(m1);
    result.mult(m2);
    return result;
  }

  static scale(m: Matrix, num: number) {
    const result = new Matrix(m.rows, m.cols);
    result.addMatrix(m);
    result.scale(num);
    return result;
  }

  static transpose(m: Matrix) {
    const result = new Matrix(m.cols, m.rows);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.vals[i][j] = m.vals[j][i];
      }
    }
    return result;
  }
}
