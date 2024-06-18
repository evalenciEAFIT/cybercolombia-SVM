#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include "Datos.h"
#include "SVMomp.h"


// Colores para la salida
const std::string rojo = "\033[41;30m";
const std::string verde = "\033[42;30m";
const std::string reset = "\033[0m";

void ingresarYClasificar(SVMomp& svm, const std::vector<std::string>& etiquetas);
void parsear_argumentos(int argc, char* argv[], std::string& fuente_datos, float& tasa_aprendizaje, float& overfitting, int& epocas, bool& echo);
 
int main(int argc, char* argv[]) {
    // Variables para almacenar los valores de los argumentos y valores por defecto 
    std::string fuente_datos;  //dataser con datos de conocimiento
    float tasa_aprendizaje = 0.001; //control de la velocidad de aprendizaje de modelo
    float overfitting = 0.001; //parámetro de regularización
    int epocas = 5;  //iteraciones del modelo
    bool echo = false;  //Bandera para ver o no el entrenamiento

    try {
        // Parsear argumentos de línea de comandos
        parsear_argumentos(argc, argv, fuente_datos, tasa_aprendizaje, overfitting, epocas, echo);

        // Imprimir información del programa
        std::cout << "Ejecución del programa...\n\n"; 
        std::cout << "DATOS DE LA INSTANCIA DEL PROGRAMA" << "\n----------------------------------\n";
        std::cout << " - Fuente de datos: " << fuente_datos << std::endl;
        std::cout << " - Tasa de aprendizaje: " << tasa_aprendizaje << std::endl;
        std::cout << " - Overfitting: " << overfitting << std::endl;
        std::cout << " - Epocas: " << epocas << std::endl;
        std::cout << " - Echo: " << (echo ? "true" : "false") << std::endl << std::endl;

        Datos lector;
        std::vector<Punto> puntos = lector.leerDatosCSV(fuente_datos);

        // Verificar que se leyeron datos correctamente
        if (puntos.empty() || lector.etiquetas.empty()) {
            std::cerr << rojo << "No se pudieron leer los datos del archivo CSV." << reset << std::endl;
            return 1;
        }

        // Crear el modelo SVM
        //float tasaAprendizaje = 0.001; //control de la velocidad de aprendizaje de modelo
        //float overfitting = 0.01; //parámetro de regularización
        int numCaracteristicas = puntos[0].caracteristicas.size(); //cantidad de variables o atributos que describen la instancia en un conjunto de datos.
        SVMomp svm(tasa_aprendizaje, overfitting, numCaracteristicas);

        // Entrenar el modelo SVM
        svm.entrenar(puntos, epocas, echo);

        // Mostrar los pesos y el sesgo del hiperplano separador
        svm.imprimirParametros(lector.etiquetas, echo);

        // Ingresar y clasificar un nuevo punto de datos
        ingresarYClasificar(svm, lector.etiquetas);

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

//=============================================================================
void ingresarYClasificar(SVMomp& svm, const std::vector<std::string>& etiquetas) {
    std::cout << "\nPRUEBAS\n" << "-------\n";

    bool otraClasificacion = true;
    while (otraClasificacion) {
        // Ingresar las características del nuevo punto
        std::cout << "Ingrese las características del nuevo punto:" << std::endl;
        std::vector<float> nuevasCaracteristicas(etiquetas.size());

        for (size_t i = 0; i < etiquetas.size(); ++i) {
            std::cout << "  " << etiquetas[i] << ": ";
            while (!(std::cin >> nuevasCaracteristicas[i])) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "         " << "\033[41;30m" << "Valor inválido. (Intente de nuevo!)"<< "\033[0m " << std::endl << "  " << etiquetas[i] << ": ";
            }
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
        std::cout << "\033[0m";

        // Preguntar si el usuario desea otra clasificación
        char otraClasificacionChar;
        std::cout << "\n¿Desea otra clasificación? ("<<verde<<" [si] s "<<reset<<" - "<<rojo<<" [no] n "<<reset<<"): ";
        std::cin >> otraClasificacionChar;

        // Convertir la respuesta del usuario a un valor booleano
        otraClasificacion = !(otraClasificacionChar == 'n' || otraClasificacionChar == 'N');
    }
}

//============================================================================================================
// Función para parsear argumentos de línea de comandos
void parsear_argumentos(int argc, char* argv[], std::string& fuente_datos, float& tasa_aprendizaje, float& overfitting, int& epocas, bool& echo) {
  // Variables para almacenar argumentos
  std::string argumento;

  // Procesar cada argumento
  for (int i = 1; i < argc; ++i) {
    argumento = argv[i];

    // Fuente de datos
    if (argumento == "--fuente_datos") {
      if (i + 1 < argc) {
        fuente_datos = argv[i + 1];
        ++i;
      } else {
        throw std::runtime_error("Falta el argumento para --fuente_datos");
      }
    }

    // Tasa de aprendizaje
    else if (argumento == "--tasa_aprendizaje") {
      if (i + 1 < argc) {
        try {
          tasa_aprendizaje = std::stof(argv[i + 1]);
        } catch (const std::exception& e) {
          throw std::runtime_error("Valor inválido para --tasa_aprendizaje: " + std::string(e.what()));
        }
        ++i;
      } else {
        throw std::runtime_error("Falta el argumento para --tasa_aprendizaje");
      }
    }

    // Overfitting
    else if (argumento == "--overfitting") {
      if (i + 1 < argc) {
        try {
          overfitting = std::stof(argv[i + 1]);
        } catch (const std::exception& e) {
          throw std::runtime_error("Valor inválido para --overfitting: " + std::string(e.what()));
        }
        ++i;
      } else {
        throw std::runtime_error("Falta el argumento para --overfitting");
      }
    }

    // Epocas
    else if (argumento == "--epocas") {
      if (i + 1 < argc) {
        try {
          epocas = std::stoi(argv[i + 1]);
        } catch (const std::exception& e) {
          throw std::runtime_error("Valor inválido para --epocas: " + std::string(e.what()));
        }
        ++i;
      } else {
        throw std::runtime_error("Falta el argumento para --epocas");
      }
    }

    // Echo
    else if (argumento == "--echo") {
      echo = true;
    }

    // Argumento no reconocido
    else {
      throw std::runtime_error("Argumento no reconocido: " + argumento);
    }
  }
}