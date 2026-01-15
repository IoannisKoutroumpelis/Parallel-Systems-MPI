import pandas as pd
import matplotlib.pyplot as plt
import os

FIG_DIR = "test/fig"
RESULTS_DIR = "test/results"

os.makedirs(FIG_DIR, exist_ok=True)

def plot_q3_1():
    file_path = os.path.join(RESULTS_DIR, "q3-1.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path, sep=' ')
    
    plt.figure(figsize=(10, 6))
    for n in df['Degree'].unique():
        subset = df[df['Degree'] == n]
        plt.plot(subset['Procs'], subset['Total_Time'], marker='o', label=f'Degree N={n}')
        
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (sec)')
    plt.title('Q3.1: Polynomial Multiplication Scalability')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'q3-1_time.png'))
    plt.close()
    print("Generated q3-1_time.png")

def plot_q3_2():
    file_path = os.path.join(RESULTS_DIR, "q3-2.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path, sep=' ')
    
    subset_spar = df[df['Procs'] == 4]
    if not subset_spar.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(subset_spar['Sparsity'], subset_spar['CSR_Total'], marker='o', label='CSR Total')
        plt.plot(subset_spar['Sparsity'], subset_spar['Dense_Comp'], marker='x', linestyle='--', label='Dense Comp')
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