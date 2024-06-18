#include "SVM.h"

#include <iostream>


SVM::SVM(float learningRate, float regularizationParam, int numCaracteristicas)
    : learningRate(learningRate), regularizationParam(regularizationParam), bias(0.0) {
    weights = std::vector<float>(numCaracteristicas, 0.0); // Inicializar pesos para las características
}

void SVM::entrenar(const std::vector<Punto>& puntos, int epochs) {
    {
        for (int i = 0; i < puntos.size(); ++i) {
            const auto& punto = puntos[i];
            // Calcular x.w + b
            float decision = 0.0;
            for (size_t j = 0; j < punto.caracteristicas.size(); ++j) {
                decision += punto.caracteristicas[j] * weights[j];
            }
            decision += bias;

            // Verificar si se viola la condición de margen
            if (punto.clase * decision <= 0) {
                {
                    // Actualizar pesos y bias
                    for (size_t j = 0; j < weights.size(); ++j) {
                        weights[j] += learningRate * (punto.clase * punto.caracteristicas[j] - regularizationParam * weights[j]);
                    }
                    bias += learningRate * punto.clase;
                }
            }
        }
    }
}

int SVM::clasificar(const Punto& puntoNuevo) const {
    float decision = 0.0;
    for (size_t i = 0; i < puntoNuevo.caracteristicas.size(); ++i) {
        decision += puntoNuevo.caracteristicas[i] * weights[i];
    }
    decision += bias;
    return (decision >= 0) ? 1 : -1;
}

void SVM::imprimirParametros(const std::vector<std::string>& etiquetas) const {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "w" << i << " {" << etiquetas[i] << "}: " << weights[i] << std::endl;
    }
    std::cout << "B: " << bias << std::endl;
}
