#ifndef DATOS_H
#define DATOS_H

#include <vector>
#include <string>
#include "Punto.h"

class Datos {
public:
    std::vector<std::string> etiquetas;
    std::vector<Punto> leerDatosCSV(const std::string& archivoCSV);
};

#endif
