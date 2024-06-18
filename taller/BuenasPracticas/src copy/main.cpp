#include <iostream>
#include <vector>
#include <string>
#include "Datos.h"
#include "SVM.h"

void ingresarYClasificar(SVM& svm, const std::vector<std::string>& etiquetas);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <archivo_datos.csv>" << std::endl;
        return 1;
    }

    std::string archivoCSV = argv[1];
    Datos lector;
    std::vector<Punto> puntos = lector.leerDatosCSV(archivoCSV);

    // Verificar que se leyeron datos correctamente
    if (puntos.empty() || lector.etiquetas.empty()) {
        std::cerr << "No se pudieron leer los datos del archivo CSV." << std::endl;
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

void ingresarYClasificar(SVM& svm, const std::vector<std::string>& etiquetas) {
    std::vector<float> nuevasCaracteristicas(etiquetas.size());
    std::cout << "Ingrese las características del nuevo punto:" << std::endl;
    for (size_t i = 0; i < etiquetas.size(); ++i) {
        std::cout << "w" << i << " {" << etiquetas[i] << "}: ";
        std::cin >> nuevasCaracteristicas[i];
    }
    Punto puntoNuevo(nuevasCaracteristicas, 0); // La clase es irrelevante para la predicción
    int clasePredicha = svm.clasificar(puntoNuevo);

    // Colores para la salida
    const std::string rojo = "\033[41;30m";
    const std::string verde = "\033[42;30m";
    const std::string reset = "\033[0m";

    if (clasePredicha == 1) {
        std::cout << "Clase predicha para el nuevo punto: " << rojo << "Maligno" << reset << std::endl;
    } else {
        std::cout << "Clase predicha para el nuevo punto: " << verde << "Benigno" << reset << std::endl;
    }
}
