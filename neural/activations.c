#include "activations.h"

#include <math.h>
#include "../matrix/ops.h"


double sigmoid(double input){
    return 1.0 / (1 + exp(-1 * input));
}

Matrix* sigmoidPrime(Matrix* m){
    Matrix* oncs = matrix_create(m->rows, m->cols);
    matrix_fill(ones, 1);
    Matrix* subtracted = subtract(once, m);
    Matrix* multiplied = multiply(m, subtracted);
    matrix_free(oncs);
    matrix_free(subtracted);
    return multiplied;
}

Matrix* softmax(Matrix* m){
    double total = 0;
    for (int i=0; i< m->rows, i++){
        for (int j=0;j< m->cols; j++){
            total += exp(m->entries[i][j]);
        }
    }
    Matrix* mat = matrix_create(m->rows, m->cols);
    for (int i=0; i< m->rows, i++){
        for (int j=0;j< m->cols; j++){
            mat->entries[i][j] = exp(m->entries[i][j]) / total;
        }
    }
    return mat;
}
