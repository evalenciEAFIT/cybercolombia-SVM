// g++ -o xyz R3_svm02.cpp -fopenmp

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>
#include <omp.h>

using namespace std;

// Clase para representar un punto de datos
class Punto {
public:
    vector<float> caracteristicas;
    int clase;

    Punto(const vector<float>& caracteristicas, int clase)
        : caracteristicas(caracteristicas), clase(clase) {}
};

// Clase para manejar la lectura de datos CSV
class Datos {
public:
    vector<string> etiquetas;

    vector<Punto> leerDatosCSV(const string& archivoCSV) {
        vector<Punto> puntos;
        ifstream archivo(archivoCSV);

        if (archivo.is_open()) {
            string linea;
            // Leer la primera línea (encabezado) y almacenar las etiquetas
            getline(archivo, linea);
            stringstream headerStream(linea);
            string etiqueta;
            getline(headerStream, etiqueta, ','); // id
            getline(headerStream, etiqueta, ','); // diagnosis
            while (getline(headerStream, etiqueta, ',')) {
                etiquetas.push_back(etiqueta);
            }

            // Leer los datos
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
};

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
        #pragma omp parallel default(none) shared(puntos, epochs, weights, bias, learningRate, regularizationParam)
        //Directiva #pragma omp parallel: 
        //Esta directiva crea un equipo de subprocesos (threads) y define una región paralela. 
        //Cada hilo en el equipo ejecutará el bloque de código dentro de esta región paralela.
        //
        //default(none) para forzar la especificación explícita de las variables compartidas y privadas, 
        //shared para especificar las variables compartidas
        {
            //#pragma omp for schedule(dynamic)
            //schedule(dynamic) para especificar la distribución dinámica de la carga de trabajo entre los hilos.
            
            //for (int epoch = 0; epoch < epochs; ++epoch) {
                //#pragma omp critical
            //    {
                    // Muestra el progreso del entrenamiento
            //        cout << "Epoch " << epoch + 1 << " de " << epochs << endl;
            //    }

                #pragma omp for
                //Directiva #pragma omp for: 
                //Esta directiva se utiliza para distribuir un bucle entre los hilos en la región paralela. 
                //Cada hilo ejecutará una porción del bucle de forma concurrente.
                for (int i = 0; i < puntos.size(); ++i) {
                    const auto& punto = puntos[i];
                    // Calcular x.w + b
                    float decision = 0.0;
                    #pragma omp simd reduction(+:decision)
                    //Directiva #pragma omp simd reduction(+:variable): 
                    //Esta directiva se utiliza para vectorizar el bucle y realizar la reducción de la variable especificada. 
                    //En el código, se utiliza para calcular la suma de las decisiones en paralelo.
                    for (size_t j = 0; j < punto.caracteristicas.size(); ++j) {
                        decision += punto.caracteristicas[j] * weights[j];
                    }
                    decision += bias;

                    // Verificar si se viola la condición de margen
                    if (punto.clase * decision <= 0) {
                        #pragma omp critical
                        //Directiva #pragma omp critical: 
                        //Esta directiva se utiliza para proteger una sección crítica del código, asegurando que solo un hilo pueda ejecutarla a la vez. 
                        //En el código proporcionado, se utiliza para garantizar que la impresión de la información de progreso del entrenamiento (cout) se realice de forma segura, evitando la escritura simultánea de múltiples hilos en la salida estándar.
                        {
                            // Actualizar pesos y bias
                            for (size_t j = 0; j < weights.size(); ++j) {
                                weights[j] += learningRate * (punto.clase * punto.caracteristicas[j] - regularizationParam * weights[j]);
                            }
                            bias += learningRate * punto.clase;
                        }
                    }
                }
            //}
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
    void imprimirParametros(const vector<string>& etiquetas) const {
        for (size_t i = 0; i < weights.size(); ++i) {
            //std::cout << etiquetas[i] << ": " << weights[i] << std::endl;
            std::cout << "w"<< i << " {" << etiquetas[i] << "}: " << weights[i] << std::endl;
        }
        std::cout << "B: " << bias << std::endl;
    }
};

// Función para ingresar y clasificar un nuevo punto de datos
void ingresarYClasificar(SVM& svm, const vector<string>& etiquetas) {
    vector<float> nuevasCaracteristicas(etiquetas.size());
    std::cout << "Ingrese las características del nuevo punto:" << std::endl;
    for (size_t i = 0; i < etiquetas.size(); ++i) {
        std::cout << "w" << i << " {" << etiquetas[i] << "}: ";
        std::cin >> nuevasCaracteristicas[i];
    }
    Punto puntoNuevo(nuevasCaracteristicas, 0); // La clase es irrelevante para la predicción
    int clasePredicha = svm.clasificar(puntoNuevo);

    // Colores para la salida
    const string rojo = "\033[41;30m";
    const string verde = "\033[42;30m";
    const string reset = "\033[0m";

    if (clasePredicha == 1) {
        std::cout << "Clase predicha para el nuevo punto: " << rojo << "Maligno" << reset << std::endl;
    } else {
        std::cout << "Clase predicha para el nuevo punto: " << verde << "Benigno" << reset << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <archivo_datos.csv>" << endl;
        return 1;
    }

    string archivoCSV = argv[1];
    Datos lector;
    vector<Punto> puntos = lector.leerDatosCSV(archivoCSV);

    // Verificar que se leyeron datos correctamente
    if (puntos.empty() || lector.etiquetas.empty()) {
        cerr << "No se pudieron leer los datos del archivo CSV." << endl;
        return 1;
    }

    // Crear el modelo SVM
    SVM svm(0.01, 0.01, puntos[0].caracteristicas.size());

    // Entrenar el modelo SVM
    svm.entrenar(puntos, 100);

    // Mostrar los pesos y el sesgo del hiperplano separador
    svm.imprimirParametros(lector.etiquetas);

    // Ingresar y clasificar un nuevo punto de datos
    ingresarYClasificar(svm, lector.etiquetas);

    return 0;
}
