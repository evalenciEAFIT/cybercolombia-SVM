#include <stdio.h>
#include <omp.h>

int main() {
  int i;

  #pragma omp parallel for
  for (i = 0; i < 10; i++) {
    printf("Hola desde el hilo %d\n", omp_get_thread_num());
  }

  return 0;
}
