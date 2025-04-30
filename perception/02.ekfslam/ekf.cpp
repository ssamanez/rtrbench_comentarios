/** $lic$
 * Copyright (c) 2022 Carnegie Mellon University
 *
 * This file is part of RTRBench.
 *
 * Permission is hereby granted, free of charge, to any
 * person obtaining a copy of this software and
 * associated documentation files (the "Software"), to
 * deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice
 * shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
 * ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
 * LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
 * EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

#include "ekf.h"
#include <vector>
#include "rtrbench_utils.h"
#include <iostream>
#include <cstdio>
#include "debug_config.h"

ExtKalmanFillter::ExtKalmanFillter(double sigX2, double sigY2, \
        double sigAlpha2, double sigBeta2, double sigR2, \
        std::vector<double> initialMeasurement) {
    this->numLandmarks = static_cast<int>(initialMeasurement.size()) / 2;

    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Número de landmarks: %d\n", this->numLandmarks);
    #endif

    this->controlCov = new double*[this->STATE_ENTRIES];
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        this->controlCov[i] = new double[this->STATE_ENTRIES]();
    }

    // DEBUG: Depuracion Matriz controlCov
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Matriz controlCov inicializada (%dx%d)\n", this->STATE_ENTRIES, this->STATE_ENTRIES);
    #endif

    this->controlCov[0][0] = sigX2;
    this->controlCov[1][1] = sigY2;
    this->controlCov[2][2] = sigAlpha2;

    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Contenido de controlCov:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.5f ", this->controlCov[i][j]);
        }
    printf("\n");
    }
    #endif

    // DEBUG: Depuracion Matriz controlCov

    this->measurementCov = new double*[this->NUM_MEAS_TYPE];
    for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
        this->measurementCov[i] = new double[this->NUM_MEAS_TYPE]();
    }
    this->measurementCov[0][0] = sigBeta2;
    this->measurementCov[1][1] = sigR2;

    // DEBUG: Mostrar contenido de measurementCov
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("measurementCov matriz después de la asignación:\n");
    for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
        for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
            printf("measurementCov[%d][%d] = %f\n", i, j, this->measurementCov[i][j]);
        }
    }
    #endif


    // Initial pos and uncertainty
    double rX = 0, rY = 0, rTheta = 0;
    double *pos = new double[this->STATE_ENTRIES];
    pos[0] = rX;
    pos[1] = rY;
    pos[2] = rTheta;

    // DEBUG: Mostrar el valor de pos
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Valores de pos después de la asignación:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        printf("pos[%d] = %f\n", i, pos[i]);
    }
    #endif


    double **posCov = new double*[this->STATE_ENTRIES];
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        posCov[i] = new double[this->STATE_ENTRIES]();
    }
    posCov[0][0] = 0.02 * 0.02;
    posCov[1][1] = 0.02 * 0.02;
    posCov[2][2] = 0.1 * 0.1;

    // DEBUG: Mostrar contenido de posCov
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Matriz posCov después de la asignación:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("posCov[%d][%d] = %f\n", i, j, posCov[i][j]);
        }
    }
    #endif


    // Landmarks
    double *landmarks = new double[2*this->numLandmarks]();
    double **landmarksCov = new double*[2*this->numLandmarks];
    for (int i = 0; i < 2*this->numLandmarks; i++) {
        landmarksCov[i] = new double[2*this->numLandmarks]();
    }

    for (int l = 0; l < this->numLandmarks; l++) {
        double lBeta = initialMeasurement[2*l];
        double lRange = initialMeasurement[2*l + 1];

        double lX = rX + lRange * cos(rTheta + lBeta);
        double lY = rY + lRange * sin(rTheta + lBeta);

        landmarks[2*l] = lX;
        landmarks[2*l + 1] = lY;

        // DEBUG: Mostrar el valor de lX y lY calculados
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Landmark %d: lX = %f, lY = %f\n", l, lX, lY);
        #endif


        double **jacobian = new double*[this->NUM_MEAS_TYPE];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            jacobian[i] = new double[this->NUM_MEAS_TYPE];
        }
        jacobian[0][0] = -lRange * sin(rTheta + lBeta);
        jacobian[0][1] = cos(rTheta + lBeta);
        jacobian[1][0] = lRange * cos(rTheta + lBeta);
        jacobian[1][1] = sin(rTheta + lBeta);

        // DEBUG: Mostrar la matriz Jacobiana
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Jacobian para Landmark %d:\n", l);
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("jacobian[%d][%d] = %f\n", i, j, jacobian[i][j]);
            }
        }
        #endif

        double **jacobianT = new double*[this->NUM_MEAS_TYPE];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            jacobianT[i] = new double[this->NUM_MEAS_TYPE];
        }
        matrixTranspose(jacobian, jacobianT, this->NUM_MEAS_TYPE, \
                this->NUM_MEAS_TYPE);

        // DEBUG: Mostrar la matriz transpuesta Jacobiana
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Transpuesta Jacobiana para Landmark %d:\n", l);
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("jacobianT[%d][%d] = %f\n", i, j, jacobianT[i][j]);
            }
        }
        #endif


        double **temp = new double*[this->NUM_MEAS_TYPE];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            temp[i] = new double[this->NUM_MEAS_TYPE];
        }
        matrixMultiplication(jacobian, this->measurementCov, temp, \
                this->NUM_MEAS_TYPE, this->NUM_MEAS_TYPE, this->NUM_MEAS_TYPE);

        // DEBUG: Mostrar el resultado de la multiplicación jacobian * measurementCov
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Matriz Temp para Landmark %d (jacobian * measurementCov):\n", l);
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("temp[%d][%d] = %f\n", i, j, temp[i][j]);
            }
        }
        #endif



        double **lCov = new double*[this->NUM_MEAS_TYPE];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            lCov[i] = new double[this->NUM_MEAS_TYPE];
        }
        matrixMultiplication(temp, jacobianT, lCov, this->NUM_MEAS_TYPE, \
                this->NUM_MEAS_TYPE, this->NUM_MEAS_TYPE);

        // DEBUG: Mostrar la matriz de covarianza de landmark lCov
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Matriz de Covarianza lCov para Landmark %d:\n", l);
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("lCov[%d][%d] = %f\n", i, j, lCov[i][j]);
            }
        }
        #endif


        landmarksCov[2*l][2*l] = lCov[0][0];
        landmarksCov[2*l][2*l+1] = lCov[0][1];
        landmarksCov[2*l+1][2*l] = lCov[1][0];
        landmarksCov[2*l+1][2*l+1] = lCov[1][1];

        // DEBUG: Mostrar los valores actualizados en landmarksCov
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Updated landmarksCov for Landmark %d:endif\n", l);
        printf("landmarksCov[%d][%d] = %f\n", 2*l, 2*l, landmarksCov[2*l][2*l]);
        printf("landmarksCov[%d][%d] = %f\n", 2*l, 2*l+1, landmarksCov[2*l][2*l+1]);
        printf("landmarksCov[%d][%d] = %f\n", 2*l+1, 2*l, landmarksCov[2*l+1][2*l]);
        printf("landmarksCov[%d][%d] = %f\n", 2*l+1, 2*l+1, landmarksCov[2*l+1][2*l+1]);
        #endif

        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] jacobian[i];
        delete[] jacobian;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] jacobianT[i];
        delete[] jacobianT;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] temp[i];
        delete[] temp;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] lCov[i];
        delete[] lCov;
    }

    // State vector X: pos and landmarks
    // Covarianec matrix P: pos and landmarks covariances
    this->numEntries = this->STATE_ENTRIES + 2*this->numLandmarks;
    this->X = new double[this->numEntries];
    this->X[0] = pos[0];
    this->X[1] = pos[1];
    this->X[2] = pos[2];

    // DEBUG: Mostrar el vector de estado X para la posición
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Vector de estado X (Posición): X[0] = %f, X[1] = %f, X[2] = %f\n", this->X[0], this->X[1], this->X[2]);
    #endif


    for (int i = 0; i < this->numLandmarks; i++) {
        this->X[this->STATE_ENTRIES + 2*i] = landmarks[2*i];
        this->X[this->STATE_ENTRIES + 2*i + 1] = landmarks[2*i + 1];

        // DEBUG: Mostrar cada landmark añadido al vector X
        #ifdef PRINT_DEBUG_CONSTRUCTOR
        printf("Landmark %d: X[%d] = %f, X[%d] = %f\n", i, this->STATE_ENTRIES + 2*i, this->X[this->STATE_ENTRIES + 2*i], this->STATE_ENTRIES + 2*i + 1, this->X[this->STATE_ENTRIES + 2*i + 1]);
        #endif

    }

    this->P = new double*[this->numEntries];
    for (int i = 0; i < this->numEntries; i++) {
        this->P[i] = new double[this->numEntries]();
    }

    // DEBUG: Mostrar la matriz P inicial (toda en ceros)
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Matriz de covarianza inicial P (zeros):\n");
    for (int i = 0; i < this->numEntries; i++) {
        for (int j = 0; j < this->numEntries; j++) {
            printf("P[%d][%d] = %f\n", i, j, this->P[i][j]);
        }
    }
    #endif



    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            this->P[i][j] = posCov[i][j];
        }
    }

    // DEBUG: Mostrar las covarianzas de la posición en la matriz P
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Covarianza de posición en P:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("P[%d][%d] = %f\n", i, j, this->P[i][j]);
        }
    }
    #endif


    for (int i = 0; i < 2*this->numLandmarks; i++) {
        for (int j = 0; j < 2*this->numLandmarks; j++) {
            int xIdx = this->STATE_ENTRIES + i;
            int yIdx = this->STATE_ENTRIES + j;
            this->P[xIdx][yIdx] = landmarksCov[i][j];
        }
    }

    // DEBUG: Mostrar las covarianzas de los landmarks en la matriz P
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Covarianzas de referencia en P:\n");
    for (int i = 0; i < 2*this->numLandmarks; i++) {
        for (int j = 0; j < 2*this->numLandmarks; j++) {
            printf("P[%d][%d] = %f\n", this->STATE_ENTRIES + i, this->STATE_ENTRIES + j, this->P[this->STATE_ENTRIES + i][this->STATE_ENTRIES + j]);
        }
    }
    #endif


    this->eye = new double*[this->numEntries];
    for (int i = 0; i < this->numEntries; i++) {
        this->eye[i] = new double[this->numEntries]();
    }
    for (int i = 0; i < this->numEntries; i++) this->eye[i][i] = 1.0;

    // DEBUG: Mostrar la matriz identidad 'eye'
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Identity Matrix 'eye':\n");
    for (int i = 0; i < this->numEntries; i++) {
        for (int j = 0; j < this->numEntries; j++) {
            printf("eye[%d][%d] = %f\n", i, j, this->eye[i][j]);
        }
    }
    #endif


    // DEBUG
    #ifdef PRINT_DEBUG_CONSTRUCTOR
    printf("Matriz P después de constructor:\n");
    for (int i = 0; i < this->numEntries; i++) {
        for (int j = 0; j < this->numEntries; j++) {
            printf("%.6f ", this->P[i][j]);
        }
        printf("\n");
    }
    #endif


    delete[] pos;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] posCov[i];
    delete[] posCov;
    delete[] landmarks;
    for (int i = 0; i < 2*this->numLandmarks; i++) delete[] landmarksCov[i];
    delete[] landmarksCov;
}

ExtKalmanFillter::~ExtKalmanFillter() {
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        delete[] this->controlCov[i];
    }
    delete[] this->controlCov;

    for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
        delete[] this->measurementCov[i];
    }
    delete[] this->measurementCov;

    delete[] this->X;

    for (int i = 0; i < this->numEntries; i++) {
        delete[] this->P[i];
    }
    delete[] P;

    for (int i = 0; i < this->numEntries; i++) {
        delete[] this->eye[i];
    }
    delete[] this->eye;
}

void ExtKalmanFillter::predict(double d, double alpha) {

    #ifdef PRINT_DEBUG_PREDICT
    printf("Entrando a la funcion predict\n");
    #endif

    double xT = this->X[0], yT = this->X[1], thetaT = this->X[2];
    // DEBUG 
    #ifdef PRINT_DEBUG_PREDICT
    printf("Estado antes: x=%.4f, y=%.4f, theta=%.4f\n", xT, yT, thetaT);
    #endif

    this->X[0] = xT + d*cos(thetaT);
    this->X[1] = yT + d*sin(thetaT);
    this->X[2] = thetaT + alpha;

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Estado después: x=%.4f, y=%.4f, theta=%.4f\n", this->X[0], this->X[1], this->X[2]);
    #endif

    double **G = new double*[this->STATE_ENTRIES];
    double **GT = new double*[this->STATE_ENTRIES];
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        G[i] = new double[this->STATE_ENTRIES];
        GT[i] = new double[this->STATE_ENTRIES];
    }
    G[0][0] = 1;
    G[0][1] = 0;
    G[0][2] = -d * sin(thetaT);
    G[1][0] = 0;
    G[1][1] = 1;
    G[1][2] = d * cos(thetaT);
    G[2][0] = 0;
    G[2][1] = 0;
    G[2][2] = 1;

    // DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Jacobian G:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", G[i][j]);
        }
        printf("\n");
    }
    #endif

    matrixTranspose(G, GT, this->STATE_ENTRIES, this->STATE_ENTRIES);

    // DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Transpuesta Jacobian GT:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", GT[i][j]);
        }
        printf("\n");
    }
    #endif


    double **L = new double*[this->STATE_ENTRIES];
    double **LT = new double*[this->STATE_ENTRIES];
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        L[i] = new double[this->STATE_ENTRIES];
        LT[i] = new double[this->STATE_ENTRIES];
    }
    L[0][0] = cos(thetaT);
    L[0][1] = -sin(thetaT);
    L[0][2] = 0;
    L[1][0] = sin(thetaT);
    L[1][1] = cos(thetaT);
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = 0;
    L[2][2] = 1;

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Jacobian L:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", L[i][j]);
        }
        printf("\n");
    }
    #endif


    matrixTranspose(L, LT, this->STATE_ENTRIES, this->STATE_ENTRIES);

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Transpuesta Jacobian LT:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", LT[i][j]);
        }
        printf("\n");
    }
    #endif


    double **temp1 = new double*[this->STATE_ENTRIES];
    double **temp2 = new double*[this->STATE_ENTRIES];
    double **temp3 = new double*[this->STATE_ENTRIES];
    double **temp4 = new double*[this->STATE_ENTRIES];
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        temp1[i] = new double[this->STATE_ENTRIES];
        temp2[i] = new double[this->STATE_ENTRIES];
        temp3[i] = new double[this->STATE_ENTRIES];
        temp4[i] = new double[this->STATE_ENTRIES];
    }

    #ifdef PRINT_DEBUG_PREDICT
    printf("Matriz P antes de matrix_mult temp1:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.6f ", this->P[i][j]);
        }
        printf("\n");
    }
    #endif


    matrixMultiplication(G, this->P, temp1, this->STATE_ENTRIES, \
            this->STATE_ENTRIES, this->STATE_ENTRIES);

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Temp1:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", temp1[i][j]);
        }
        printf("\n");
    }
    #endif


    matrixMultiplication(temp1, GT, temp2, this->STATE_ENTRIES, \
            this->STATE_ENTRIES, this->STATE_ENTRIES);

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Temp2:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", temp2[i][j]);
        }
        printf("\n");
    }
    #endif


    matrixMultiplication(L, this->controlCov, temp3, this->STATE_ENTRIES, \
            this->STATE_ENTRIES, this->STATE_ENTRIES);

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Temp3:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", temp3[i][j]);
        }
        printf("\n");
    }
    #endif

    
    matrixMultiplication(temp3, LT, temp4, this->STATE_ENTRIES, \
            this->STATE_ENTRIES, this->STATE_ENTRIES);

    //DEBUG
    #ifdef PRINT_DEBUG_PREDICT
    printf("Temp4:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.4f ", temp4[i][j]);
        }
        printf("\n");
    }
    #endif


    matrixAddition(temp2, temp4, this->P, this->STATE_ENTRIES, \
            this->STATE_ENTRIES);

    #ifdef PRINT_DEBUG_PREDICT
    printf("Matriz P después de predicción:\n");
    for (int i = 0; i < this->STATE_ENTRIES; i++) {
        for (int j = 0; j < this->STATE_ENTRIES; j++) {
            printf("%.6f ", this->P[i][j]);
        }
        printf("\n");
    }
    #endif



    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] G[i];
    delete[] G;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] GT[i];
    delete[] GT;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] L[i];
    delete[] L;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] LT[i];
    delete[] LT;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] temp1[i];
    delete[] temp1;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] temp2[i];
    delete[] temp2;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] temp3[i];
    delete[] temp3;
    for (int i = 0; i < this->STATE_ENTRIES; i++) delete[] temp4[i];
    delete[] temp4;
}

void ExtKalmanFillter::update(const std::vector<double> &measure) {

    #ifdef PRINT_DEBUG_UPDATE
    printf("\n--- UPDATE ---\n");
    #endif

    assert(this->numLandmarks * 2 == static_cast<int>(measure.size()));

    for (int l = 0; l < this->numLandmarks; l++) {
        double xT = this->X[0], yT = this->X[1], thetaT = this->X[2];
        double lX = this->X[this->STATE_ENTRIES + 2*l];
        double lY = this->X[this->STATE_ENTRIES + 2*l + 1];

        double q = (lX-xT)*(lX-xT) + (lY-yT)*(lY-yT);
        double qSqrt = sqrt(q);

        //DEBUG
        #ifdef PRINT_DEBUG_UPDATE
        printf("\n--- Landmark %d ---\n", l);
        printf("Robot pos: (%.6f, %.6f, %.6f)\n", xT, yT, thetaT);
        printf("Landmark pos: (%.6f, %.6f)\n", lX, lY);
        printf("q = %.6f, sqrt(q) = %.6f\n", q, qSqrt);
        #endif


        double **H = new double*[this->NUM_MEAS_TYPE];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            H[i] = new double[this->numEntries]();
        }
        H[0][0] = (lY - yT) / q;
        H[0][1] = (-lX + xT) / q;
        H[0][2] = -1.0;
        H[1][0] = (-lX + xT) / qSqrt;
        H[1][1] = (-lY + yT) / qSqrt;
        H[1][2] = 0.0;

        H[0][this->STATE_ENTRIES + 2*l] = (-lY + yT) / q;
        H[0][this->STATE_ENTRIES + 2*l + 1] = (lX - xT) / q;
        H[1][this->STATE_ENTRIES + 2*l] = (lX - xT) / qSqrt;
        H[1][this->STATE_ENTRIES + 2*l + 1] = (lY - yT) / qSqrt;

        //DEBUG
        #ifdef PRINT_DEBUG_UPDATE
        printf("H matrix:\n");
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                printf("%.6f ", H[i][j]);
            }
            printf("\n");
        }
        #endif


        double **HT = new double*[this->numEntries];
        for (int i = 0; i < this->numEntries; i++) {
            HT[i] = new double[this->NUM_MEAS_TYPE];
        }
        matrixTranspose(H, HT, this->NUM_MEAS_TYPE, this->numEntries);

        printf("numEntries: %i - NUM_MEAS_TYPE: %i \n", this->numEntries, this->NUM_MEAS_TYPE);

        //DEBUG
        #ifdef PRINT_DEBUG_UPDATE
        printf("HT matrix:\n");
        for (int i = 0; i <  this->numEntries; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("%.6lf - Indice i %i - Indice j %i \n", HT[i][j], i, j);
            }
            printf("\n");
        }
        #endif


        double expectedBeta = wrapToPi(atan2(lY-yT, lX-xT) - thetaT);
        double expectedRange = qSqrt;

        double lBeta = measure[2*l];
        double lRange = measure[2*l + 1];

        //DEBUG
        #ifdef PRINT_DEBUG_UPDATE
        printf("Expected (beta, range): (%.6f, %.6f)\n", expectedBeta, expectedRange);
        printf("Measured (beta, range): (%.6f, %.6f)\n", lBeta, lRange);
        #endif


        double **diff = new double*[this->NUM_MEAS_TYPE];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) diff[i] = new double[1];
        diff[0][0] = lBeta - expectedBeta;
        diff[1][0] = lRange - expectedRange;

        //DEBUG
        #ifdef PRINT_DEBUG_UPDATE
        printf("\n--- Diff vector ---\n");
        printf("diff[0][0] = %.6f\n", diff[0][0]);
        printf("diff[1][0] = %.6f\n", diff[1][0]);
        #endif


        double **temp1 = new double*[this->NUM_MEAS_TYPE];
        double **temp2 = new double*[this->NUM_MEAS_TYPE];
        double **temp3 = new double*[this->NUM_MEAS_TYPE];
        double **temp4 = new double*[this->NUM_MEAS_TYPE];
        double **temp5 = new double*[this->numEntries];
        double **kalmanGain = new double*[this->numEntries];
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            temp1[i] = new double[this->numEntries];
            temp2[i] = new double[this->NUM_MEAS_TYPE];
            temp3[i] = new double[this->NUM_MEAS_TYPE];
            temp4[i] = new double[this->NUM_MEAS_TYPE];
        }
        for (int i = 0; i < this->numEntries; i++) {
            temp5[i] = new double[this->NUM_MEAS_TYPE];
            kalmanGain[i] = new double[this->NUM_MEAS_TYPE];
        }

        matrixMultiplication(H, this->P, temp1, this->NUM_MEAS_TYPE, \
                this->numEntries, this->numEntries);

        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp1:\n");
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                printf("%.4f ", temp1[i][j]);
            }
            printf("\n");
        }
        #endif

    
        matrixMultiplication(temp1, HT, temp2, this->NUM_MEAS_TYPE, \
                this->numEntries, this->NUM_MEAS_TYPE);

        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp2:\n");
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("%.4f ", temp2[i][j]);
            }
            printf("\n");
        }
        #endif

                
        matrixAddition(temp2, this->measurementCov, temp3, \
                this->NUM_MEAS_TYPE, this->NUM_MEAS_TYPE);

        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp3:\n");
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("%.4f ", temp3[i][j]);
            }
            printf("\n");
        }
        #endif


                
        matrix2dInverse(temp3, temp4);

        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp4:\n");
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("%.4f ", temp4[i][j]);
            }
            printf("\n");
        }
        #endif


        matrixMultiplication(this->P, HT, temp5, this->numEntries, \
                this->numEntries, this->NUM_MEAS_TYPE);

        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp5:\n");
        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("%.4f ", temp5[i][j]);
            }
            printf("\n");
        }
        #endif


        matrixMultiplication(temp5, temp4, kalmanGain, this->numEntries, \
                this->NUM_MEAS_TYPE, this->NUM_MEAS_TYPE);
                
        #ifdef PRINT_DEBUG_UPDATE
        printf("\n--- Kalman Gain ---\n");
        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->NUM_MEAS_TYPE; j++) {
                printf("K[%d][%d] = %.6f\n", i, j, kalmanGain[i][j]);
            }
        }
        #endif


        double **stateUpdate = new double*[this->numEntries];
        for (int i = 0; i < this->numEntries; i++) {
            stateUpdate[i] = new double[1];
        }
        matrixMultiplication(kalmanGain, diff, stateUpdate, this->numEntries, \
                this->NUM_MEAS_TYPE, 1);
        
        #ifdef PRINT_DEBUG_UPDATE
        printf("\n--- State Update ---\n");
        for (int i = 0; i < this->numEntries; i++) {
            printf("ΔX[%d] = %.6f\n", i, stateUpdate[i][0]);
            this->X[i] += stateUpdate[i][0];
        }
        #endif


        /*
        printf("\n--- State Update ---\n");
        for (int i = 0; i < this->numEntries; i++) {
            printf("ΔX[%d] = %.6f\n", i, stateUpdate[i][0]);
            this->X[i] += stateUpdate[i][0];
        }
        */

        double **covUpdate = new double*[this->numEntries];
        double **temp6 = new double*[this->numEntries];
        double **temp7 = new double*[this->numEntries];
        for (int i = 0; i < this->numEntries; i++) {
            covUpdate[i] = new double[this->numEntries];
            temp6[i] = new double[this->numEntries];
            temp7[i] = new double[this->numEntries];
        }

        matrixMultiplication(kalmanGain, H, temp6, this->numEntries, \
                this->NUM_MEAS_TYPE, this->numEntries);
        
        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp6:\n");
        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                printf("%.4f ", temp6[i][j]);
            }
            printf("\n");
        }
        #endif


        matrixSubtraction(this->eye, temp6, covUpdate, this->numEntries, \
                this->numEntries);
                
        #ifdef PRINT_DEBUG_UPDATE
        printf("CovUpdate:\n");
        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                printf("%.4f ", covUpdate[i][j]);
            }
            printf("\n");
        }
        #endif


        matrixMultiplication(covUpdate, this->P, temp7, this->numEntries, \
                this->numEntries, this->numEntries);

        #ifdef PRINT_DEBUG_UPDATE
        printf("Temp7:\n");
        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                printf("%.4f ", temp7[i][j]);
            }
            printf("\n");
        }
        #endif


        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                this->P[i][j] = temp7[i][j];
            }
        }

        // DEBUG
        #ifdef PRINT_DEBUG_UPDATE
        printf("Matriz P después de update:\n");
        for (int i = 0; i < this->numEntries; i++) {
            for (int j = 0; j < this->numEntries; j++) {
                printf("%.6f ", this->P[i][j]);
            }
            printf("\n");
        }
        #endif


        // -O3 merges most of the following loops.
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] H[i];
        delete[] H;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] diff[i];
        delete[] diff;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] temp1[i];
        delete[] temp1;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] temp2[i];
        delete[] temp2;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] temp3[i];
        delete[] temp3;
        for (int i = 0; i < this->NUM_MEAS_TYPE; i++) delete[] temp4[i];
        delete[] temp4;
        for (int i = 0; i < this->numEntries; i++) delete[] temp5[i];
        delete[] temp5;
        for (int i = 0; i < this->numEntries; i++) delete[] temp6[i];
        delete[] temp6;
        for (int i = 0; i < this->numEntries; i++) delete[] temp7[i];
        delete[] temp7;
        for (int i = 0; i < this->numEntries; i++) delete[] HT[i];
        delete[] HT;
        for (int i = 0; i < this->numEntries; i++) delete[] kalmanGain[i];
        delete[] kalmanGain;
        for (int i = 0; i < this->numEntries; i++) delete[] stateUpdate[i];
        delete[] stateUpdate;
        for (int i = 0; i < this->numEntries; i++) delete[] covUpdate[i];
        delete[] covUpdate;
    }
}

std::vector<double> ExtKalmanFillter::getState() const {
    return std::vector<double>(this->X, this->X + this->numEntries);
}
