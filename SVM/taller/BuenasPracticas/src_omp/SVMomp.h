#ifndef SVMomp_H
#define SVMomp_H

#include <vector>
#include <string>


#include "Punto.h"

class SVMomp {
private:
    std::vector<float> weights;
    float bias;
    float learningRate;
    float regularizationParam;

public:
    SVMomp(float learningRate, float regularizationParam, int numCaracteristicas);

    void entrenar(const std::vector<Punto>& puntos, int epochs, bool mostrarProgreso = false);
    int clasificar(const Punto& puntoNuevo) const;
    void imprimirParametros(const std::vector<std::string>& etiquetas, bool mostarResumen = false) const;
};

#endif
