#ifndef PUNTO_H
#define PUNTO_H

#include <vector>

class Punto {
public:
    std::vector<float> caracteristicas;
    int clase;

    Punto(const std::vector<float>& caracteristicas, int clase)
        : caracteristicas(caracteristicas), clase(clase) {}
};

#endif
