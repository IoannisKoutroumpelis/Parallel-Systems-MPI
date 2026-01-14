#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Initialize array with random non-zero integers [1, 10]
void init_random_poly(int *poly, int size) {
    for (int i = 0; i < size; i++) {
        poly[i] = (rand() % 10) + 1;
    }
}

// Check for memory allocation failure
void check_malloc(void *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    int n, coeffs_n, result_n;
    
    // Pointers for process 0
    int *global_A = NULL;
    int *global_B = NULL;
    long long *global_C = NULL;

    // Arrays for MPI Scatterv parameters
    int *sendcounts = NULL;
    int *displs = NULL;

    // Timers
    double t_start_comm, t_send_duration = 0.0;
    double t_start_comp, t_comp_duration = 0.0;
    double t_start_recv, t_recv_duration = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Argument validation
    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <degree n>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    n = atoi(argv[1]);
    coeffs_n = n + 1;       // A degree n poly has n+1 terms
    result_n = 2 * n + 1;   // Multiplication results in degree 2n


    if (rank == 0) {
        global_A = (int *)malloc(coeffs_n * sizeof(int));
        global_B = (int *)malloc(coeffs_n * sizeof(int));
        global_C = (long long *)calloc(result_n, sizeof(long long));
        
        check_malloc(global_A);
        check_malloc(global_B);
        check_malloc(global_C);

        srand(time(NULL));
        init_random_poly(global_A, coeffs_n);
        init_random_poly(global_B, coeffs_n);

        // Prepare Scatterv arrays (handling uneven distribution)
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        check_malloc(sendcounts);
        check_malloc(displs);

        int remainder = coeffs_n % size;
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (coeffs_n / size) + (i < remainder ? 1 : 0);
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

   
    // Calculate local count and displacement for each process independently
    int remainder = coeffs_n % size;
    int local_count = (coeffs_n / size) + (rank < remainder ? 1 : 0);
    int global_offset = rank * (coeffs_n / size) + (rank < remainder ? rank : remainder);

    // Allocate local buffers
    int *local_A = (int *)malloc(local_count * sizeof(int));
    int *local_B = (int *)malloc(coeffs_n * sizeof(int)); // All procs need full B
    long long *local_C = (long long *)calloc(result_n, sizeof(long long)); // Temp result
    
    check_malloc(local_A);
    check_malloc(local_B);
    check_malloc(local_C);

    // Sync before timing communication
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_start_comm = MPI_Wtime();

    // Scatter polynomial A
    MPI_Scatterv(global_A, sendcounts, displs, MPI_INT, 
                 local_A, local_count, MPI_INT, 
                 0, MPI_COMM_WORLD);

    // For Bcast, root needs to copy global_B to local_B pointer or use MPI_IN_PLACE logic.
    // Simplest approach: Root copies data to local buffer before Bcast.
    if (rank == 0) {
        for(int i = 0; i < coeffs_n; i++) local_B[i] = global_B[i];
    }

    // Broadcast polynomial B
    MPI_Bcast(local_B, coeffs_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) t_send_duration = MPI_Wtime() - t_start_comm;

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all data is received
    if (rank == 0) t_start_comp = MPI_Wtime();

    // Convolution: local_A[i] corresponds to x^(global_offset + i)
    // local_B[j] corresponds to x^j
    // Product contributes to x^(global_offset + i + j)
    for (int i = 0; i < local_count; i++) {
        int global_idx_A = global_offset + i;
        for (int j = 0; j < coeffs_n; j++) {
            local_C[global_idx_A + j] += (long long)local_A[i] * local_B[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_comp_duration = MPI_Wtime() - t_start_comp;

    if (rank == 0) t_start_recv = MPI_Wtime();

    // Sum partial polynomials from all processes into global_C
    MPI_Reduce(local_C, global_C, result_n, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        t_recv_duration = MPI_Wtime() - t_start_recv;
        
        // Output performance metrics
        printf("Polynomial Degree N: %d | MPI Processes: %d\n", n, size);
        printf("-------------------------------------------\n");
        printf("(i)   Data Distr Time:  %f sec\n", t_send_duration);
        printf("(ii)  Computation Time: %f sec\n", t_comp_duration);
        printf("(iii) Result Coll Time: %f sec\n", t_recv_duration);
        printf("(iv)  Total Exec Time:  %f sec\n", 
               t_send_duration + t_comp_duration + t_recv_duration);
        printf("-------------------------------------------\n");

        // Cleanup Root resources
        free(global_A);
        free(global_B);
        free(global_C);
        free(sendcounts);
        free(displs);
    }

    // Cleanup Local resources
    free(local_A);
    free(local_B);
    free(local_C);

    MPI_Finalize();
    return EXIT_SUCCESS;
}