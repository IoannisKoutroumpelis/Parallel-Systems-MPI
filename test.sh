#!/bin/bash

BIN_DIR="bin"
RESULTS_DIR="test/results"

mkdir -p $RESULTS_DIR

OUTPUT_FILE="$RESULTS_DIR/q3-1.txt"
echo "Running Q3-1 Experiments..."
echo "Degree Procs Total_Time" > $OUTPUT_FILE

DEGREES=(10000 50000)
PROCS=(1 2 4 8)

for n in "${DEGREES[@]}"; do
    for p in "${PROCS[@]}"; do
        echo "  Running Degree=$n with NP=$p..."
        TIME=$(mpirun -np $p ./$BIN_DIR/q3-1 $n | grep "(iv)" | awk '{print $5}')
        
        if [ ! -z "$TIME" ]; then
            echo "$n $p $TIME" >> $OUTPUT_FILE
        else
            echo "Error running N=$n P=$p"
        fi
    done
done

OUTPUT_FILE="$RESULTS_DIR/q3-2.txt"
echo "Running Q3-2 Experiments..."
echo "Size Sparsity Iterations Procs CSR_Total Dense_Comp" > $OUTPUT_FILE

SIZE=2000
ITER=20
SPARSITY_VALS=(0.0 0.5 0.9 0.99)
PROCS=(1 2 4 8)

for spar in "${SPARSITY_VALS[@]}"; do
    for p in "${PROCS[@]}"; do
        echo "  Running Size=$SIZE Sparsity=$spar NP=$p..."
        
        OUT=$(mpirun -np $p ./$BIN_DIR/q3-2 $SIZE $spar $ITER)
        
        CSR_TIME=$(echo "$OUT" | grep "(iv)" | awk '{print $5}')
        DENSE_TIME=$(echo "$OUT" | grep "(v)" | awk '{print $5}')
        
        if [ ! -z "$CSR_TIME" ]; then
            echo "$SIZE $spar $ITER $p $CSR_TIME $DENSE_TIME" >> $OUTPUT_FILE
        fi
    done
done

echo "Done. Results saved in $RESULTS_DIR"