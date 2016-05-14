#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#define USE_LIKWID_PERFCTR

#ifdef USE_LIKWID_PERFCTR
#include <likwid.h>
#endif

/* Valor absoluto de um ponto flutuante */
#define FABS(x)       ((x < 0.0) ? (-x) : (x))

/* Tamanho do desenrolamento de laços */
#define UNROLL_SIZE     (4)

/* Obtém do tempo atual os segundos e milisegundos */
double timestamp() {
  struct timeval tp;

  gettimeofday(&tp, NULL);

  return ((double)(tp.tv_sec + tp.tv_usec / 1000000.0));
}

/* Gera sistemas lineares aleatórios positivos definidos simétricos */
int generateRandomPositiveDefiniteLinearSystem(unsigned int N, double *A, double *b) {
  if(!A || !b) {
    return -1;
  }

  /* generate a randomly initialized matrix in row-major order */
  double *ptr = A;
  double *end = A + N * N;

  double invRandMax = 1.0 / (double)RAND_MAX;

  while(ptr != end) {
    *ptr++ = (double)rand() * invRandMax;
  }

  /*  Now we want to make this matrix positive definite. Since all
    values are positive and <1.0, we have to ensure it is symmetric
    and diagonally dominant.

    A = A + transpose(A)
    A = A + I*N            */

  unsigned int i, j;
  for(i = 0; i < N; ++i) {
    for(j = i + 1; j < N; ++j) {
      double aux = A[i * N + j];
      A[i * N + j] += A[j * N + i];
      A[j * N + i] += aux;
    }
  }

  for(i = 0; i < N; ++i) {
    A[i * N + i] += A[i * N + i] + N;
  }

  /* create the vector of independent terms (b) */
  for(i = 0; i < N; ++i) {
    b[i] = (double)rand() * invRandMax;
  }

  return 0;
}

/* Lê sistemas lineares da entrada */
void readLinearSystem(FILE *in, unsigned int n, double *A, double *b, unsigned int end_offset) {
  unsigned int i, j;

  for(i = 0; i < n - end_offset; ++i) {
    for(j = 0; j < n; ++j) {
      fscanf(in, "%lf", A + i * n + j);
    }
  }

  for(i = 0; i < n; ++i) {
    fscanf(in, "%lf", b + i);
  }
}

/* Imprime um vetor */
void printVector(FILE *out, double *v, unsigned int n) {
  unsigned int i;

  for(i = 0; i < n; ++i) {
    fprintf(out, "%.17g ", v[i]);
  }

  fprintf(out, "\n");
}

/* Imprime um sistema linear */
void printLinearSystem(FILE *out, double *A, double *b, unsigned int n, unsigned int end_offset) {
  unsigned int i, j;

  for(i = 0; i < n - end_offset; ++i) {
    for(j = 0; j < n; ++j) {
      fprintf(out, "%.17g(x%d) ", A[i * n + j], j);
    }

    fprintf(out, "= %.17g\n", b[i]);
  }
}

/* Verifica se n é potência de 2 */
unsigned char isPowerOf2(unsigned int n) {
  unsigned int setbits = 0;

  while(n != 0) {
    if(n & 0x1) {
      ++setbits;
    }

    n >>= 1;
  }

  return (setbits == 1) ? 1 : 0;
}

/* Calcula a raiz quadrada até certa precisão usando o método de Newton Raphson */
double sqrt_prec(double value, double error) {
  double x, nx;

  if(FABS(value) < error) {
    return 0.0;
  }

  nx = value / 2;

  do {
    x = nx;
    nx = x - ((x * x - value) / (2 * x));
  } while(FABS((nx - x) / nx) > error);

  return nx;
}

/* Calcula o residuo 'Ax - b' e retorna seu módulo, a norma é armazenada em 'e'
 * e o tempo gasto em segundos é armazenado em 'terror', m é um valor real que
 * irá multiplicar todos os elementos de A durante o cálculo do produto */
double calcResidue(double *res, double *A, double *x, double *b, double m, unsigned int n, double error, double *e, double *terror, unsigned int end_offset) {
  double *Ai;
  double prod, sum = 0.0, errorts, iter[UNROLL_SIZE];
  unsigned int i, j;

#ifdef USE_LIKWID_PERFCTR
  likwid_markerStartRegion("Residue");
#endif

  errorts = timestamp();

  for(i = 0; i < n; ++i) {
    prod = 0.0;

    /* Obtem a matriz A na linha i */
    Ai = A + (i * (n + end_offset));

    /* Loop unrolling */
    for(j = 0; j < n - (n % UNROLL_SIZE); j += UNROLL_SIZE) {
      iter[0] = Ai[j] * x[j];
      iter[1] = Ai[j + 1] * x[j + 1];
      iter[2] = Ai[j + 2] * x[j + 2];
      iter[3] = Ai[j + 3] * x[j + 3];
      prod += m * (iter[0] + iter[1] + iter[2] + iter[3]);
    }

    /* Calcula para os elementos que faltaram */
    for(j = n - (n % UNROLL_SIZE); j < n; ++j) {
      prod += m * Ai[j] * x[j];
    }

    res[i] = b[i] - prod;
    sum += res[i] * res[i];
  }

  *e = sqrt(sum);
  *terror = timestamp() - errorts;

#ifdef USE_LIKWID_PERFCTR
  likwid_markerStopRegion("Residue");
#endif

  return sum;
}

/* Calcula a solução Ax = b utilizando o método do gradiente, onde maxiter é o 
 * número máximo de iterações que podem ser executadas, 'error' contém o erro
 * máximo aceito e recebe o erro final da operação, e os tempos gastos em
 * segundos para o cálculo do gradiente e do residuo são armazenados em
 * 'tgrad' e 'terror', respectivamente */
void gradSolver(unsigned int n, double *A, double *b, double *x, unsigned int maxiter, double *error, double *tgrad, double *terror, unsigned int end_offset) {
  double *dir, *res, *Aj;
  double alpha, beta, mod, aux, e, iter[UNROLL_SIZE];
  unsigned int i, j, k;

  /* Aloca espaço para os vetores de direção e resíduo */
  if((dir = (double *) malloc(n * 2 * sizeof(double))) == NULL) {
    fprintf(stderr, "Erro ao alocar memória!\n");
    return;
  }

  *tgrad = timestamp();
  *terror = 0.0;
  memset(x, 0, n * sizeof(double));

  res = dir + n;
  mod = calcResidue(res, A, x, b, 1.0, n, *error, &e, terror, end_offset);
  memcpy(dir, res, n * sizeof(double));

#ifdef USE_LIKWID_PERFCTR
  likwid_markerInit();
  likwid_markerStartRegion("Lambda");
#endif

  /* Enquanto não atingir o número máximo de iterações e o erro for menor do que o aceito */
  for(i = 0; i < maxiter && e > *error; ++i) {
    /* Calcula aux = dir'.A.dir */
    aux = 0.0;
    for(j = 0; j < n; ++j) {
      /* Obtem os valores de A na linha j */
      Aj = A + (j * (n + end_offset));

      /* Loop unrolling */
      for(k = 0; k < j - (j % UNROLL_SIZE); k += UNROLL_SIZE) {
        iter[0] = dir[k] * Aj[k];
        iter[1] = dir[k + 1] * Aj[k + 1];
        iter[2] = dir[k + 2] * Aj[k + 2];
        iter[3] = dir[k + 3] * Aj[k + 3];
        aux += 2 * dir[j] * (iter[0] + iter[1] + iter[2] + iter[3]);
      }

      /* Calcula para os elementos que faltaram */
      for(k = j - (j % UNROLL_SIZE); k < j; ++k) {
        aux += 2 * dir[j] * dir[k] * Aj[k];
      }

      aux += dir[j] * dir[j] * Aj[j];
    }

    /* Calcula alpha */
    alpha = mod / aux;

    /* Calcula x = x + alpha.dir */
    /* Loop unrolling (usa vetorização SIMD) */
    for(j = 0; j < n - (n % UNROLL_SIZE); j += UNROLL_SIZE) {
      x[j] = x[j] + dir[j] * alpha;
      x[j + 1] = x[j + 1] + dir[j + 1] * alpha;
      x[j + 2] = x[j + 2] + dir[j + 2] * alpha;
      x[j + 3] = x[j + 3] + dir[j + 3] * alpha;
    }

    /* Calcula para os elementos que faltaram */
    for(j = n - (n % UNROLL_SIZE); j < n; ++j) {
      x[j] = x[j] + dir[j] * alpha;
    }

    /* Salva o módulo da iteração anterior */
    aux = mod;

    /* Calcula novo residuo alpha.A.dir - res e armazena seu módulo em mod */
    mod = calcResidue(res, A, dir, res, alpha, n, *error, &e, terror, end_offset);

    /* Calcula beta */
    beta = mod / aux;

    /* Calcula dir = res + beta.dir */
    /* Loop unrolling (usa vetorização SIMD) */
    for(j = 0; j < n - (n % UNROLL_SIZE); j += UNROLL_SIZE) {
      dir[j] = res[j] + dir[j] * beta;
      dir[j + 1] = res[j + 1] + dir[j + 1] * beta;
      dir[j + 2] = res[j + 2] + dir[j + 2] * beta;
      dir[j + 3] = res[j + 3] + dir[j + 3] * beta;
    }

    /* Calcula para os elementos que faltaram */
    for(j = n - (n % UNROLL_SIZE); j < n; ++j) {
      dir[j] = res[j] + dir[j] * beta;
    }

  }

#ifdef USE_LIKWID_PERFCTR
  likwid_markerStopRegion("Lambda");
  likwid_markerClose();
#endif

  /* Divide os tempos tgrad e terror pelo número de iterações */
  *tgrad = (timestamp() - *tgrad - *terror) / (i + 1);
  *terror = *terror / (i + 1);

  *error = e;
  free(dir);
}

/* Função principal */
int main(int argc, char *const *argv) {
  FILE *in = stdin;
  FILE *out = stdout;
  double *A, *b, *x;
  double error = 0.0001;
  double tgrad, terror;
  unsigned char gen = 0, end_offset = 0;
  unsigned int maxIter = 0;
  unsigned int n;
  int opt;

  /* Define a semente para gerar números aleatórios */
  srand(20142);

  /* Lê as opções passadas ao programa */
  while((opt = getopt(argc, argv, "i:o:r:k:e:")) != -1) {
    switch(opt) {
      /* Arquivo de entrada */
      case 'i':
        in = fopen(optarg, "r");

        if(in == NULL) {
          fprintf(stderr, "Falha ao abrir o arquivo de entrada.\n");
          return -2;
        }

        break;

      /* Arquivo de saída */
      case 'o':
        out = fopen(optarg, "w");

        if(out == NULL) {
          fprintf(stderr, "Falha ao abrir o arquivo de saida.\n");
          return -3;
        }

        break;

      /* Tamanho para gerar matrizes aleatórias */
      case 'r':
        n = atoi(optarg);
        gen = 1;
        break;

      /* Número máximo de iterações */
      case 'k':
        maxIter = atoi(optarg);
        break;

      /* Erro máximo permitido */
      case 'e':
        error = atof(optarg);
        break;

      default:
        fprintf(stderr, "Uso: %s [-i arquivo_entrada] [-o arquivo_saida] [-r N] [-k maxIter] [-e erro]\n", argv[0]);
        return 0;
    }
  }

  /* Se for ler a matriz (e não gerar), lê o tamanho no arquivo de entrada */
  if(gen == 0) {
    fscanf(in, "%d", &n);
  }

  if(n > 0) {
    /* Se não definido, o número máximo de iterações é 2*n */
    if(!maxIter) {
      maxIter = n * 2;
    }

    /* Aloca a matriz A e os vetores x e b */
    if(isPowerOf2(n) == 1) {
      end_offset = 1;
    }

    A = (double *) malloc(((n * (n + end_offset)) + (2 * n)) * sizeof(double));

    if(A == NULL) {
      fprintf(stderr, "Erro ao alocar memória!\n");
      return -1;
    }

    /* Aponta o ponteiro b para o final do espaço alocado para A */
    b = A + n * (n + end_offset);
    /* Aponta o ponteiro x para o final do espaço alocado para B */
    x = b + n;

    /* Lê os valores de A, b e x ou os gera com valores aleatórios caso necessário */
    if(gen == 0) {
      readLinearSystem(in, n, A, b, end_offset);
    } else {
      generateRandomPositiveDefiniteLinearSystem(n + end_offset, A, b);
    }

    /* Calcula x pelo método do gradiente */
    gradSolver(n, A, b, x, maxIter, &error, &tgrad, &terror, end_offset);

    /* Imprime o resultado */
    fprintf(out,  "#\n"
            "# Erro: %.17g\n"
            "# Tempo Grad: %.17g\n"
            "# Tempo Erro: %.17g\n"
            "#\n"
            "%d\n",
            error, tgrad, terror, n);

    printVector(out, x, n);

    /* Libera a memória alocada por A, x e b */
    free(A);
  }

  if(in != stdin) {
    fclose(in);
  }

  if(out != stdout) {
    fclose(out);
  }

  return 0;
}
