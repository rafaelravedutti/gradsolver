# Compilador a ser usado
CC=gcc
# Arquivos fonte
SOURCES=gradSolver.c
# Nome do executável
EXEC=gradSolver
# Opções do compilador
CFLAGS=-Wall -I. -lm -llikwid -lpthread -DLIKWID_PERFMON -O0

all: $(EXEC)

$(EXEC): $(SOURCES)
	$(CC) $(SOURCES) $(CFLAGS) -o $(EXEC)

clean:
	@echo "> Removendo arquivos..."
	rm -f $(EXEC)
