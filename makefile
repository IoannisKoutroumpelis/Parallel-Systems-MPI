CC = mpicc
CFLAGS = -O3 -Wall -Wextra
BIN_DIR = bin
SRC_DIR = src

# Τα ονόματα των εκτελέσιμων (με παύλες)
TARGETS = q3-1 q3-2

# Οι πλήρεις διαδρομές των εκτελέσιμων (bin/q3-1, bin/q3-2)
EXECS = $(patsubst %,$(BIN_DIR)/%,$(TARGETS))

all: $(BIN_DIR) $(EXECS)

# Δημιουργία του φακέλου bin αν δεν υπάρχει
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Κανόνας για το q3-1
$(BIN_DIR)/q3-1: $(SRC_DIR)/q3-1.c
	$(CC) $(CFLAGS) -o $@ $<

# Κανόνας για το q3-2
$(BIN_DIR)/q3-2: $(SRC_DIR)/q3-2.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean