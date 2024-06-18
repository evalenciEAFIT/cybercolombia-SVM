#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>

using namespace std;

// Clase para representar un punto de datos
class Punto {
public:
    float x;   // característica 1
    float y;   // característica 2
    int clase; // clasificación

    Punto(float x, float y, int clase) : x(x), y(y), clase(clase) {}
};

// Función para leer puntos de datos de un archivo CSV y obtener nombres de características
vector<Punto> leerDatosCSV(const string& archivoCSV, vector<string>& nombresCaracteristicas) {
    vector<Punto> puntos;
    ifstream archivo(archivoCSV);

    if (archivo.is_open()) {
        string linea;
        // Leer la primera línea (encabezado)
        if (getline(archivo, linea)) {
            stringstream ss(linea);
            string nombre;
            while (getline(ss, nombre, ',')) {
                nombresCaracteristicas.push_back(nombre);
            }
        }
        // Leer los datos
        while (getline(archivo, linea)) {
            float x, y;
            int clase;

            // Leer valores de la línea CSV
            sscanf(linea.c_str(), "%f,%f,%d", &x, &y, &clase);

            // Crear un nuevo punto de datos y agregarlo al vector
            puntos.emplace_back(x, y, clase);
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
    vector<float> weights;     // pesos por características
    float bias;                // sesgo
    float learningRate;        // tasa de aprendizaje
    float regularizationParam; // parámetro de regularización
    vector<string> nombresCaracteristicas; // nombres de las características

public:
    SVM(float learningRate, float regularizationParam, const vector<string>& nombresCaracteristicas)
        : learningRate(learningRate), regularizationParam(regularizationParam), bias(0.0), nombresCaracteristicas(nombresCaracteristicas) {
        weights = vector<float>(2, 0.0); // Inicializar pesos para dos características
    }

    // Función para entrenar el clasificador SVM
    void entrenar(const vector<Punto>& puntos, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& punto : puntos) {
                // Calcular x.w + b
                float decision = punto.x * weights[0] + punto.y * weights[1] + bias;

                // Verificar si se viola la condición de margen
                if (punto.clase * decision <= 0) {
                    // Actualizar pesos y bias
                    weights[0] += learningRate * (punto.clase * punto.x - regularizationParam * weights[0]);
                    weights[1] += learningRate * (punto.clase * punto.y - regularizationParam * weights[1]);
                    bias += learningRate * punto.clase;
                }
            }
        }
    }

    // Función para clasificar un nuevo punto de datos
    int clasificar(const Punto& puntoNuevo) const {
        float decision = puntoNuevo.x * weights[0] + puntoNuevo.y * weights[1] + bias;
        return (decision >= 0) ? 1 : -1;
    }

    // Función para imprimir los pesos y el sesgo (bias)
    void imprimirParametros() const {
        cout << "w0 {" << nombresCaracteristicas[0] << "}: " << weights[0] << endl;
        cout << "w1 {" << nombresCaracteristicas[1] << "}: " << weights[1] << endl;
        cout << "b: " << bias << endl;
    }
};

int main() {
    // Vector para almacenar los nombres de las características
    vector<string> nombresCaracteristicas;

    // Leer datos de un archivo CSV y obtener nombres de características
    vector<Punto> puntos = leerDatosCSV("datos.csv", nombresCaracteristicas);

    // Crear el modelo SVM
    SVM svm(0.01, 0.01, nombresCaracteristicas);

    // Entrenar el modelo SVM
    svm.entrenar(puntos, 100);

    // Mostrar los pesos y el sesgo del hiperplano separador
    svm.imprimirParametros();

    // Clasificar un nuevo punto de datos
    Punto puntoNuevo(2.8, 0.8, 0); // El tercer valor es irrelevante para la predicción
    int clasePredicha = svm.clasificar(puntoNuevo);

    cout << "Clase predicha para el nuevo punto: " << clasePredicha << endl;

    return 0;
}
