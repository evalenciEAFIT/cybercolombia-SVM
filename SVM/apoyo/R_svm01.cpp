#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

// Clase para representar un punto de datos
class Punto {
public:
    vector<float> caracteristicas;
    int clase;

    Punto(const vector<float>& caracteristicas, int clase) 
        : caracteristicas(caracteristicas), clase(clase) {}
};

// Función para leer puntos de datos de un archivo CSV
vector<Punto> leerDatosCSV(const string& archivoCSV) {
    vector<Punto> puntos;
    ifstream archivo(archivoCSV);

    if (archivo.is_open()) {
        string linea;
        // Ignorar la primera línea (encabezado)
        getline(archivo, linea);

        while (getline(archivo, linea)) {
            stringstream lineStream(linea);
            string cell;
            vector<float> caracteristicas;
            int clase;

            // Leer id y diagnóstico
            getline(lineStream, cell, ','); // id
            getline(lineStream, cell, ','); // diagnóstico
            clase = (cell == "M") ? 1 : -1;

            // Leer las características
            while (getline(lineStream, cell, ',')) {
                caracteristicas.push_back(stof(cell));
            }

            // Crear un nuevo punto de datos y agregarlo al vector
            puntos.emplace_back(caracteristicas, clase);
        }
        archivo.close();
    } else {
        cerr << "Error al abrir el archivo CSV: " << archivoCSV << endl;
    }

    return puntos;
}

// Clase para la implementación de SVM
class SVM {
private:
    vector<float> weights;
    float bias;
    float learningRate;
    float regularizationParam;

public:
    SVM(float learningRate, float regularizationParam, int numCaracteristicas)
        : learningRate(learningRate), regularizationParam(regularizationParam), bias(0.0) {
        weights = vector<float>(numCaracteristicas, 0.0); // Inicializar pesos para las características
    }

    // Función para entrenar el clasificador SVM
    void entrenar(const vector<Punto>& puntos, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& punto : puntos) {
                // Calcular x.w + b
                float decision = 0.0;
                for (size_t i = 0; i < punto.caracteristicas.size(); ++i) {
                    decision += punto.caracteristicas[i] * weights[i];
                }
                decision += bias;

                // Verificar si se viola la condición de margen
                if (punto.clase * decision <= 0) {
                    // Actualizar pesos y bias
                    for (size_t i = 0; i < weights.size(); ++i) {
                        weights[i] += learningRate * (punto.clase * punto.caracteristicas[i] - regularizationParam * weights[i]);
                    }
                    bias += learningRate * punto.clase;
                }
            }
        }
    }

    // Función para clasificar un nuevo punto de datos
    int clasificar(const Punto& puntoNuevo) const {
        float decision = 0.0;
        for (size_t i = 0; i < puntoNuevo.caracteristicas.size(); ++i) {
            decision += puntoNuevo.caracteristicas[i] * weights[i];
        }
        decision += bias;
        return (decision >= 0) ? 1 : -1;
    }

    // Función para imprimir los pesos y el bias
    void imprimirParametros() const {
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << "w" << i << ": " << weights[i] << std::endl;
        }
        std::cout << "b: " << bias << std::endl;
    }
};

int main() {
    // Leer datos de un archivo CSV
    vector<Punto> puntos = leerDatosCSV("datos_completos.csv");

    // Crear el modelo SVM
    SVM svm(0.01, 0.01, puntos[0].caracteristicas.size());

    // Entrenar el modelo SVM
    svm.entrenar(puntos, 100);

    // Mostrar los pesos y el sesgo del hiperplano separador
    svm.imprimirParametros();

    // Clasificar un nuevo punto de datos
    //vector<float> nuevasCaracteristicas = { 14.5, 20.2, 90.5,  0.5,   1.2,    1.3,    1.5,    1.6,   1.2,    1.1,   0.7,   1.1,  2.2, 2.3,     1.3,    0.4,     1.4,   1.2,    1.4,     1.1,  0.4,  1.7,  2.0,  2.1,  2.2,    1.3,    1.5,    1.7,  1.2,    0.9}; // Ejemplo de características
    //vector<float> nuevasCaracteristicas = {9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773};
    vector<float> nuevasCaracteristicas = {15.1,22.02,97.26,712.8,0.09056,0.07081,0.05253,0.03334,0.1616,0.05684,0.3105,0.8339,2.097,29.91,0.004675,0.0103,0.01603,0.009222,0.01095,0.001629,18.1,31.69,117.7,1030,0.1389,0.2057,0.2712,0.153,0.2675,0.07873};
    Punto puntoNuevo(nuevasCaracteristicas, 0); // El tercer valor es irrelevante para la predicción
    int clasePredicha = svm.clasificar(puntoNuevo);

    //std::cout << "Clase predicha para el nuevo punto: " << clasePredicha << std::endl;
    std::cout << "Clase predicha para el nuevo punto: " <<  ((clasePredicha == 1) ? "M" : "B") << std::endl;
   

    return 0;
}
