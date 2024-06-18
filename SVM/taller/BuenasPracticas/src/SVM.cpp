#include "SVM.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <ctime>  // Para std::chrono::milliseconds
#include <sys/resource.h> // Para obtener el uso de memoria

// Secuencia de escape para colores ANSI
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

SVM::SVM(float learningRate, float regularizationParam, int numCaracteristicas)
    : learningRate(learningRate), regularizationParam(regularizationParam), bias(0.0) {
    if (numCaracteristicas <= 0) {
        throw std::invalid_argument("El número de características debe ser mayor que 0.");
    }
    weights = std::vector<float>(numCaracteristicas, 0.0); // Inicializar pesos para las características
}

void SVM::entrenar(const std::vector<Punto>& puntos, int epochs, bool mostrarProgreso) {
    if (epochs <= 0) {
        throw std::invalid_argument("El número de épocas debe ser mayor que 0.");
    }
    if (puntos.empty()) {
        throw std::invalid_argument("El conjunto de puntos no debe estar vacío.");
    }

    auto start = std::chrono::high_resolution_clock::now(); // Marca de tiempo de inicio

    if (mostrarProgreso) {
        std::cout << "ENTRENANDO EL MODELO" << "\n--------------------" << std::endl;
    }
    
    //ajustar el modelo por iteraciones
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int correct = 0; // Para contar el número de clasificaciones correctas

        //- - - - - [ OPERACIONES MATETICAS DEL MODELO ] - - - - - - - - - - 
        for (size_t i = 0; i < puntos.size(); ++i) {
            const auto& punto = puntos[i];
            // Calcular x.w + b
            float decision = 0.0;
            for (size_t j = 0; j < punto.caracteristicas.size(); ++j) {
                decision += punto.caracteristicas[j] * weights[j];
            }
            decision += bias;

            // Verificar si se viola la condición de margen
            if (punto.clase * decision <= 0) {
                // Actualizar pesos y bias
                for (size_t j = 0; j < weights.size(); ++j) {
                    weights[j] += learningRate * (punto.clase * punto.caracteristicas[j] - regularizationParam * weights[j]);
                }
                bias += learningRate * punto.clase;
            } else {
                correct++;
            }
        //- - - - - - - - - - - - - - - - - - - - - - - -  

            if (mostrarProgreso) {
                // Mostrar progreso
                int interval = puntos.size() / 10 == 0 ? 1 : puntos.size() / 10; // Evitar división por cero
                if (i % interval == 0 || i == puntos.size() - 1) { // Muestra progreso cada 10% y al final
                    int progress = static_cast<int>(std::round((i + 1) * 100.0 / puntos.size()));
                    int barWidth = 20;
                    int pos = barWidth * progress / 100;

                    // Obtener el uso de memoria
                    struct rusage usage;
                    getrusage(RUSAGE_SELF, &usage);
                    long memoryUsage = usage.ru_maxrss; // Uso máximo de memoria residente

                    std::cout << "\rEpoch " << epoch + 1 << "/" << epochs 
                              << " [";
                    for (int k = 0; k < barWidth; ++k) {
                        if (k < pos) std::cout << "=";
                        else if (k == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::setw(3) << progress << "% "
                              << "Correctos: " << correct << " de " << puntos.size()
                              << std::flush;
                }
            }
        }
        // Fin de la época
        if (mostrarProgreso) {
            std::cout << "\rEpoch " << epoch + 1 << "/" << epochs << " [====================] " << ANSI_COLOR_GREEN << "100% Correctos: " << correct << " de " << puntos.size() <<" (NO violan la condición de margen)" << ANSI_COLOR_RESET <<" { " << "Sesgo (Bias): " << bias <<" }" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now(); // Marca de tiempo de fin
    std::chrono::duration<double> duration = end - start; // Duración del entrenamiento

    // Obtener el uso de memoria final
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long memoryUsage = usage.ru_maxrss; // Uso máximo de memoria residente

    if (mostrarProgreso) {
        std::cout << "\nRESUMEN CONSUMO ENTRENAMIENTO\n";
        std::cout << " - Entrenamiento completo en " << duration.count() << " segundos." << std::endl;
        std::cout << " - Uso máximo de memoria: " << memoryUsage << " KB" << std::endl << std::endl;
    } else {
        std::cout << "Modelo entrenado " << ANSI_COLOR_GREEN <<"OK." << ANSI_COLOR_RESET << std::endl << std::endl;
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

void SVM::imprimirParametros(const std::vector<std::string>& etiquetas, bool mostarResumen) const {
    if (mostarResumen){
        std::cout << "\nRESUMEN DE LOS PARÁMETROS" << "\n----------------------------" << std::endl;
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << " w" << i << " {" << etiquetas[i] << "}: " << weights[i] << std::endl;
        }
        std::cout << " B: " << bias << std::endl << std::endl;
    } else {
        std::cout << "Modelo entrenado " << ANSI_COLOR_GREEN <<"OK." << ANSI_COLOR_RESET << std::endl << std::endl;
    }
}
