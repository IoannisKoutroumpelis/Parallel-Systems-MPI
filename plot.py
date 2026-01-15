import pandas as pd
import matplotlib.pyplot as plt
import os

FIG_DIR = "test/fig"
RESULTS_DIR = "test/results"

os.makedirs(FIG_DIR, exist_ok=True)


q1_threads = [1, 2, 4, 8]


q1_speedup_10k = [1.15, 2.05, 2.15, 3.5]
q1_speedup_50k = [1.05, 2.12, 4.25, 4.28]

def plot_q3_1():
    file_path = os.path.join(RESULTS_DIR, "q3-1.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path, sep=' ')
    
    # Plot: Speedup Comparison (MPI vs pThreads)
    
    plt.figure(figsize=(12, 7))
    
    
    for n in df['Degree'].unique():
        subset = df[df['Degree'] == n].sort_values(by='Procs')
        
        # Βρίσκουμε τον χρόνο για 1 διεργασία (Serial Time)
        serial_row = subset[subset['Procs'] == 1]
        if serial_row.empty:
            continue
        
        t_serial = serial_row['Total_Time'].values[0]
        subset['Speedup'] = t_serial / subset['Total_Time']
        
        plt.plot(subset['Procs'], subset['Speedup'], marker='o', linewidth=2, label=f'MPI (Current) N={n}')
    # Προσθέτουμε τα δεδομένα από την εικόνα σου για σύγκριση
    plt.plot(q1_threads, q1_speedup_10k, marker='x', linestyle='--', alpha=0.7, label='Assig 1 (Shared Mem) N=10000')
    plt.plot(q1_threads, q1_speedup_50k, marker='x', linestyle='--', alpha=0.7, label='Assig 1 (Shared Mem) N=50000')

    # Ideal Linear Speedup (Reference)
    plt.plot([1, 8], [1, 8], 'k:', alpha=0.5, label='Ideal Linear')

    plt.xlabel('Number of Processes / Threads')
    plt.ylabel('Speedup')
    plt.title('Comparison: MPI (Distributed) vs Assignment 1 (Shared Memory)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'q3-1_compare_speedup.png'))
    plt.close()
    print("Generated q3-1_compare_speedup.png")

def plot_q3_2():
    file_path = os.path.join(RESULTS_DIR, "q3-2.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path, sep=' ')
    
    subset_spar = df[df['Procs'] == 4]
    if not subset_spar.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(subset_spar['Sparsity'], subset_spar['CSR_Total'], marker='o', label='CSR Total (MPI)')
        plt.plot(subset_spar['Sparsity'], subset_spar['Dense_Comp'], marker='x', linestyle='--', label='Dense Comp (MPI)')
        plt.xlabel('Sparsity (0.0=Full, 1.0=Empty)')
        plt.ylabel('Time (sec)')
        plt.title('Q3.2: CSR vs Dense Performance (NP=4)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(FIG_DIR, 'q3-2_sparsity.png'))
        plt.close()
        print("Generated q3-2_sparsity.png")

if __name__ == "__main__":
    plot_q3_1()
    plot_q3_2()