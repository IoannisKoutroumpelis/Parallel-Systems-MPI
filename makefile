# Compiler settings
CC = mpicc
CFLAGS = -O3 -Wall -Wextra

# Executable names
EXEC1 = q3-1
EXEC2 = q3-2

# Source files
SRC1 = q3-1.c
SRC2 = q3-2.c

NP ?= 4

DEG ?= 10000

SIZE ?= 2000
SPAR ?= 0.9
ITER ?= 10
all: $(EXEC1) $(EXEC2)

$(EXEC1): $(SRC1)
	$(CC) $(CFLAGS) -o $(EXEC1) $(SRC1)

$(EXEC2): $(SRC2)
	$(CC) $(CFLAGS) -o $(EXEC2) $(SRC2)


run1: $(EXEC1)
	@echo "--- Running Polynomial Multiplication (q3-1) ---"
	@echo "Processes: $(NP), Degree: $(DEG)"
	mpirun -np $(NP) ./$(EXEC1) $(DEG)

run2: $(EXEC2)
	@echo "--- Running Sparse Matrix Multiplication (q3-2) ---"
	@echo "Processes: $(NP), Size: $(SIZE), Sparsity: $(SPAR), Iterations: $(ITER)"
	mpirun -np $(NP) ./$(EXEC2) $(SIZE) $(SPAR) $(ITER)


clean:
	rm -f $(EXEC1) $(EXEC2)

.PHONY: all clean run1 run2