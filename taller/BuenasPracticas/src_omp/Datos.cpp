#include "Datos.h"

#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Punto> Datos::leerDatosCSV(const std::string& archivoCSV) {
    std::vector<Punto> puntos;
    std::ifstream archivo(archivoCSV);

    if (archivo.is_open()) {
        std::string linea;
        // Leer la primera línea (encabezado) y almacenar las etiquetas
        std::getline(archivo, linea);
        std::stringstream headerStream(linea);
        std::string etiqueta;
        std::getline(headerStream, etiqueta, ','); // id
        std::getline(headerStream, etiqueta, ','); // diagnosis
        while (std::getline(headerStream, etiqueta, ',')) {
            etiquetas.push_back(etiqueta);
        }

        // Leer los datos
        while (std::getline(archivo, linea)) {
            std::stringstream lineStream(linea);
            std::string cell;
            std::vector<float> caracteristicas;
            int clase;

            // Leer id y diagnóstico
            std::getline(lineStream, cell, ','); // id
            std::getline(lineStream, cell, ','); // diagnóstico
            clase = (cell == "M") ? 1 : -1;

            // Leer las características
            while (std::getline(lineStream, cell, ',')) {
                caracteristicas.push_back(std::stof(cell));
            }

            // Crear un nuevo punto de datos y agregarlo al vector
            puntos.emplace_back(caracteristicas, clase);
        }
        archivo.close();
    } else {
        // Colores para la salida
        const std::string rojo = "\033[41;30m";
        const std::string reset = "\033[0m";

        std::cerr << rojo << "Error al abrir el archivo CSV: " << archivoCSV << reset << std::endl;
    }

    return puntos;
}
