#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Structure to represent a Matrix in CSR format
typedef struct {
    double *values;     // Non-zero values
    int *col_indices;   // Column indices for values
    int *row_ptr;       // Index in 'values' where each row starts (size N+1)
    int nnz;            // Total number of non-zero elements
    int n_rows;         // Number of rows (N)
} CSRMatrix;

// Helper to check memory allocation
void check_malloc(void *ptr) {
    if (!ptr) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

// Generate random dense matrix and vector
// Sparsity is a float between 0.0 (full) and 1.0 (empty)
void generate_data(double *A, double *x, int N, double sparsity) {
    for (int i = 0; i < N * N; i++) {
        double r = (double)rand() / RAND_MAX;
        if (r < sparsity) {
            A[i] = 0.0;
        } else {
            A[i] = ((double)(rand() % 10)) + 1.0; // Random value 1-10
        }
    }
    for (int i = 0; i < N; i++) {
        x[i] = 1.0; // Simplify vector init to 1.0
    }
}

// Convert Dense Matrix to CSR format
void dense_to_csr(double *A, int N, CSRMatrix *csr) {
    csr->n_rows = N;
    // First pass: Count non-zeros
    int nnz = 0;
    for (int i = 0; i < N * N; i++) {
        if (A[i] != 0.0) nnz++;
    }
    csr->nnz = nnz;

    csr->values = (double *)malloc(nnz * sizeof(double));
    csr->col_indices = (int *)malloc(nnz * sizeof(int));
    csr->row_ptr = (int *)malloc((N + 1) * sizeof(int));
    check_malloc(csr->values);
    check_malloc(csr->col_indices);
    check_malloc(csr->row_ptr);

    int count = 0;
    for (int i = 0; i < N; i++) {
        csr->row_ptr[i] = count;
        for (int j = 0; j < N; j++) {
            double val = A[i * N + j];
            if (val != 0.0) {
                csr->values[count] = val;
                csr->col_indices[count] = j;
                count++;
            }
        }
    }
    csr->row_ptr[N] = count;
}

void free_csr(CSRMatrix *csr) {
    if (csr->values) free(csr->values);
    if (csr->col_indices) free(csr->col_indices);
    if (csr->row_ptr) free(csr->row_ptr);
}

int main(int argc, char** argv) {
    int rank, size;
    int N, iterations;
    double sparsity_ratio;

    // Timers
    double t_csr_create_start, t_csr_create_end, t_csr_create_dur = 0;
    double t_comm_start, t_comm_end, t_comm_dur = 0;
    double t_csr_comp_start, t_csr_comp_end, t_csr_comp_dur = 0;
    double t_dense_comp_start, t_dense_comp_end, t_dense_comp_dur = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Usage: %s <N> <sparsity> <iterations>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    sparsity_ratio = atof(argv[2]);
    iterations = atoi(argv[3]);

    if (N % size != 0) {
        if (rank == 0) printf("Error: N (%d) must be divisible by process count (%d) for simplicity.\n", N, size);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_rows = N / size;
    double *global_A_dense = NULL;
    double *global_x = (double *)malloc(N * sizeof(double));
    double *local_x = (double *)malloc(N * sizeof(double)); // Holds full X (replicated)
    double *local_y = (double *)malloc(local_rows * sizeof(double)); // Holds partial result
    CSRMatrix global_csr;

    if (rank == 0) {
        srand(time(NULL));
        global_A_dense = (double *)malloc(N * N * sizeof(double));
        check_malloc(global_A_dense);
        
        generate_data(global_A_dense, global_x, N, sparsity_ratio);

        t_csr_create_start = MPI_Wtime();
        dense_to_csr(global_A_dense, N, &global_csr);
        t_csr_create_end = MPI_Wtime();
        t_csr_create_dur = t_csr_create_end - t_csr_create_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_comm_start = MPI_Wtime();

    // 1a. Broadcast initial vector x
    // In rank 0 global_x has data. We copy to local_x to broadcast.
    if (rank == 0) {
        for(int i=0; i<N; i++) local_x[i] = global_x[i];
    }
    MPI_Bcast(local_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 1b. Distribute CSR structure
    // We need to calculate how many non-zeros (nnz) each process gets.
    int *sendcounts_nnz = NULL;
    int *displs_nnz = NULL;
    int local_nnz = 0;
    
    // We need to send 'row_ptr' info. 
    // Trick: Send 'local_rows' entries of row_ptr. 
    // Each proc needs row_ptr[start...end].
    int *local_row_ptr_raw = (int*)malloc((local_rows + 1) * sizeof(int));

    if (rank == 0) {
        sendcounts_nnz = (int *)malloc(size * sizeof(int));
        displs_nnz = (int *)malloc(size * sizeof(int));
        
        // Calculate NNZ for each process based on row distribution
        for (int i = 0; i < size; i++) {
            int start_row = i * local_rows;
            int end_row = (i + 1) * local_rows;
            sendcounts_nnz[i] = global_csr.row_ptr[end_row] - global_csr.row_ptr[start_row];
            displs_nnz[i] = global_csr.row_ptr[start_row];
        }
    }

    // Scatter the NNZ count so each process knows how much to alloc
    MPI_Scatter(sendcounts_nnz, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local CSR arrays
    double *local_values = (double *)malloc(local_nnz * sizeof(double));
    int *local_cols = (int *)malloc(local_nnz * sizeof(int));
    check_malloc(local_values);
    check_malloc(local_cols);

    // Scatter variable amounts of values and column indices
    MPI_Scatterv(rank == 0 ? global_csr.values : NULL, sendcounts_nnz, displs_nnz, MPI_DOUBLE,
                 local_values, local_nnz, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
                 
    MPI_Scatterv(rank == 0 ? global_csr.col_indices : NULL, sendcounts_nnz, displs_nnz, MPI_INT,
                 local_cols, local_nnz, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Handle Row Pointers.
    // We need to send the corresponding slice of global row_ptr.
    // However, global row_ptr has cumulative indices.
    // We will Scatter the raw chunk, then normalize it locally.
    int *row_ptr_send_buf = NULL; // Temp buffer for scattering row pointers
    if (rank == 0) {
        // We only send the STARTING pointers for the rows handled by each proc.
        // We send local_rows elements. The end of the last row is inferred from local_nnz.
        row_ptr_send_buf = global_csr.row_ptr; 
        // Note: This logic assumes global_csr.row_ptr is contiguous. 
        // We send local_rows elements starting from rank*local_rows.
    }
    
    // Receive the chunk of row_ptr (size = local_rows)
    // We allocate local_row_ptr of size local_rows + 1
    int *my_row_chunk = (int*)malloc(local_rows * sizeof(int));
    MPI_Scatter(row_ptr_send_buf, local_rows, MPI_INT, 
                my_row_chunk, local_rows, MPI_INT, 
                0, MPI_COMM_WORLD);

    // Construct valid local_row_ptr (0-based for local_values)
    // The first element of my_row_chunk is the global offset for this process.
    int global_offset = my_row_chunk[0];
    for (int i = 0; i < local_rows; i++) {
        local_row_ptr_raw[i] = my_row_chunk[i] - global_offset;
    }
    // The last element (end of last row) is exactly local_nnz
    local_row_ptr_raw[local_rows] = local_nnz;
    free(my_row_chunk);

    if (rank == 0) {
        t_comm_end = MPI_Wtime();
        t_comm_dur = t_comm_end - t_comm_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_csr_comp_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; iter++) {
        // 1. Local SpMV Multiplication
        for (int i = 0; i < local_rows; i++) {
            double sum = 0.0;
            int start = local_row_ptr_raw[i];
            int end = local_row_ptr_raw[i+1];
            for (int j = start; j < end; j++) {
                sum += local_values[j] * local_x[local_cols[j]];
            }
            local_y[i] = sum;
        }

        // 2. Allgather to update vector X for next iteration
        // The result 'y' becomes 'x' for the next step.
        // Every process gets the full updated vector.
        MPI_Allgather(local_y, local_rows, MPI_DOUBLE, 
                      local_x, local_rows, MPI_DOUBLE, 
                      MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_csr_comp_end = MPI_Wtime();
        t_csr_comp_dur = t_csr_comp_end - t_csr_comp_start;
    }

   
    // Prepare dense data distribution
    double *local_dense_A = (double *)malloc(local_rows * N * sizeof(double));
    check_malloc(local_dense_A);

    // Reset vector x to initial state
    if (rank == 0) {
        for(int i=0; i<N; i++) local_x[i] = global_x[i];
    }
    MPI_Bcast(local_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter Dense Matrix
    MPI_Scatter(global_A_dense, local_rows * N, MPI_DOUBLE,
                local_dense_A, local_rows * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_dense_comp_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; iter++) {
        // 1. Local Dense MatVec
        for (int i = 0; i < local_rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += local_dense_A[i * N + j] * local_x[j];
            }
            local_y[i] = sum;
        }

        // 2. Allgather
        MPI_Allgather(local_y, local_rows, MPI_DOUBLE, 
                      local_x, local_rows, MPI_DOUBLE, 
                      MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_dense_comp_end = MPI_Wtime();
        t_dense_comp_dur = t_dense_comp_end - t_dense_comp_start;

        printf("Grid: %dx%d | Sparsity: %.2f | Iterations: %d | Procs: %d\n", 
               N, N, sparsity_ratio, iterations, size);
        printf("------------------------------------------------------------\n");
        printf("(i)   CSR Creation Time:      %f sec\n", t_csr_create_dur);
        printf("(ii)  CSR Distr Time:         %f sec\n", t_comm_dur);
        printf("(iii) CSR Computation Time:   %f sec\n", t_csr_comp_dur);
        printf("(iv)  Total CSR Time:         %f sec\n", t_csr_create_dur + t_comm_dur + t_csr_comp_dur);
        printf("(v)   Total Dense Time:       %f sec (Computation Only)\n", t_dense_comp_dur);
        printf("------------------------------------------------------------\n");
        
        free(global_A_dense);
        free(global_x);
        free_csr(&global_csr);
        free(sendcounts_nnz);
        free(displs_nnz);
    }

    free(local_values);
    free(local_cols);
    free(local_row_ptr_raw);
    free(local_x);
    free(local_y);
    free(local_dense_A);

    MPI_Finalize();
    return EXIT_SUCCESS;
}