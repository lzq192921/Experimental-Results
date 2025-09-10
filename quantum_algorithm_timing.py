import time
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import XGate
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import math

# Ensure the directory for saving images exists
FIGURE_DIR = r"D:\quantum\fig"
os.makedirs(FIGURE_DIR, exist_ok=True)  # Create directory if it doesn't exist

# 1. Define problem parameters and helper functions
def generate_i0(k):
    """Generate target index i0 based on k value"""
    if k == 2:
        return 1  # Binary representation: 01
    else:  # k >= 3
        return 5 % (2**k)  # Binary representation of 5 adapts to different k values

def g(x, n):
    """Define g(x) function: x XOR 0^(n-2)10"""
    mask = (1 << (n-1))  # (n-1)th bit is 1, others are 0
    return x ^ mask

def f(i, y, n, i0):
    """Define f(i, y) function"""
    if i == i0:
        # When i=i0, return constant 0^(n-2)01
        return 1  # Last bit is 1, others are 0
    else:
        # When i≠i0, return i_trunc XOR y, where i_trunc is the n-bit truncation of i
        i_trunc = i % (2**n)
        return i_trunc ^ y

# 2. Implement quantum algorithm Oracle
def create_f_oracle(k, n, i0):
    """Create quantum Oracle for f(i, y)"""
    # Register sizes: k bits for i, n bits for y, n bits for output f(i,y)
    oracle = QuantumCircuit(k + n + n, name="O_f")
    
    # For i≠i0 case: f(i,y) = i_trunc XOR y
    for qubit in range(n):
        oracle.cx(k + qubit, k + n + qubit)  # Copy y to output register
        
    i_trunc = i0 % (2**n)
    for qubit in range(n):
        if (i_trunc >> qubit) & 1:
            oracle.x(k + n + qubit)
    
    # For i=i0 case: f(i0,y) = 0^(n-2)01 (constant)
    i0_bin = format(i0, f'0{k}b')[::-1]  # Reverse to match qubit order
    for qubit in range(k):
        if i0_bin[qubit] == '0':
            oracle.x(qubit)
    
    # When i=i0, set output to constant 1
    oracle.x(k + n)  # Set first bit of output register to 1
    for qubit in range(1, n):
        oracle.x(k + n + qubit)  # Set other bits to 0
    
    # Undo marking
    for qubit in range(k):
        if i0_bin[qubit] == '0':
            oracle.x(qubit)
    
    # Add control to ensure operations only生效 when i=i0
    multi_control = XGate().control(num_ctrl_qubits=k)
    target_qubit = k + n  # Target qubit
    if target_qubit < k + n + n:
        oracle.append(multi_control, list(range(k)) + [target_qubit])
    else:
        raise ValueError(f"Oracle target qubit {target_qubit} is out of range")
    
    return oracle

# 3. Build complete quantum algorithm circuit
def build_quantum_algorithm(k, n, l, i0):
    """Construct the complete quantum algorithm circuit"""
    # Calculate register sizes
    index_qubits = k                     # Index register
    x_qubits = n * (l + 1)               # x_j registers
    g_qubits = n * (l + 1)               # g(x_j) registers
    f_qubits = n * (l + 1)               # f(i,g(x_j)) output registers
    b_qubit = 1                          # b register
    r_qubit = 1                          # r register
    
    # Total number of qubits
    total_qubits = index_qubits + x_qubits + g_qubits + f_qubits + b_qubit + r_qubit
    
    # Initialize circuit
    qc = QuantumCircuit(total_qubits, k)
    
    # Define starting indices for each register
    x_reg_start = index_qubits
    g_reg_start = x_reg_start + x_qubits
    f_reg_start = g_reg_start + g_qubits
    b_qubit_idx = f_reg_start + f_qubits
    r_qubit_idx = b_qubit_idx + b_qubit
    
    # Step 1: Prepare |psi_g> state - random x_j and corresponding g(x_j)
    np.random.seed(42)  # Fix seed for reproducibility
    for j in range(l+1):
        x_j = np.random.randint(0, 2**n)
        g_xj = g(x_j, n)
        
        # Encode x_j into quantum state
        for qubit in range(n):
            pos = x_reg_start + j * n + qubit
            if (x_j >> qubit) & 1:
                qc.x(pos)
        
        # Encode g(x_j) into quantum state
        for qubit in range(n):
            pos = g_reg_start + j * n + qubit
            if (g_xj >> qubit) & 1:
                qc.x(pos)
    
    # Step 2: Apply H^(k) to create uniform superposition
    qc.h(range(index_qubits))
    
    # Step 3: Prepare |b> = |-> state
    qc.x(b_qubit_idx)
    qc.h(b_qubit_idx)
    
    # Step 4: Apply Oracle O_f (l+1 times)
    oracle = create_f_oracle(k, n, i0)
    
    for j in range(l+1):
        i_reg = list(range(index_qubits))
        y_reg = [g_reg_start + j * n + qubit for qubit in range(n)]
        f_reg = [f_reg_start + j * n + qubit for qubit in range(n)]
        
        qc.append(oracle.to_instruction(), i_reg + y_reg + f_reg)
    
    # Step 5: Implement B(i) test (r register)
    for j in range(l):
        for qubit in range(n):
            ctrl_pos = f_reg_start + j * n + qubit
            targ_pos = f_reg_start + (j+1) * n + qubit
            qc.cx(ctrl_pos, targ_pos)
            qc.x(targ_pos)
    
    # Combine results into r register
    control_qubits = [f_reg_start + l * n + qubit for qubit in range(n)]
    qc.mcx(control_qubits, r_qubit_idx)
    
    # Step 6: Perform b XOR r
    qc.cx(r_qubit_idx, b_qubit_idx)
    
    # Step 7: Undo B(i) test (uncompute r)
    for j in range(l-1, -1, -1):
        for qubit in range(n):
            pos1 = f_reg_start + j * n + qubit
            pos2 = f_reg_start + (j+1) * n + qubit
            qc.x(pos2)
            qc.cx(pos1, pos2)
    
    # Step 8: Apply inverse Oracle (l+1 times)
    for j in range(l+1):
        i_reg = list(range(index_qubits))
        y_reg = [g_reg_start + j * n + qubit for qubit in range(n)]
        f_reg = [f_reg_start + j * n + qubit for qubit in range(n)]
        
        qc.append(oracle.inverse().to_instruction(), i_reg + y_reg + f_reg)
    
    # Step 9: Apply Grover iterations
    num_states = 2**k
    if num_states == 1:
        grover_iterations = 0
    else:
        grover_iterations = int(np.round(np.pi / (4 * np.arcsin(1 / np.sqrt(num_states)))))
    
    grover_iterations = max(1, grover_iterations) if num_states > 1 else 0
    print(f"k={k}, search space={num_states}, Grover iterations={grover_iterations}")
    
    # Diffusion operator
    def diffusion_operator():
        diff = QuantumCircuit(k, name="Diffusion")
        diff.h(range(k))
        diff.x(range(k))
        
        # Multi-controlled Z gate
        diff.h(k-1)
        diff.mcx(list(range(k-1)), k-1)
        diff.h(k-1)
        
        diff.x(range(k))
        diff.h(range(k))
        return diff
    
    # Apply Grover iterations
    for _ in range(grover_iterations):
        # Apply Oracle to mark target state
        i0_bin = format(i0, f'0{k}b')[::-1]  # Reverse to match qubit order
        for qubit in range(k):
            if i0_bin[qubit] == '0':
                qc.x(qubit)
        
        qc.h(k-1)
        qc.mcx(list(range(k-1)), k-1)
        qc.h(k-1)
        
        for qubit in range(k):
            if i0_bin[qubit] == '0':
                qc.x(qubit)
        
        # Apply diffusion operator
        qc.append(diffusion_operator().to_instruction(), range(k))
    
    # Measurement
    qc.measure(range(k), range(k))
    
    return qc

# 4. Performance evaluation function
def evaluate_performance(k_values, n_values, shots=1024, noise=False):
    """Evaluate algorithm performance and collect required statistics"""
    results = []
    total_start_time = time.time()  # Total time measurement start
    
    # Create noise model
    noise_model = None
    if noise:
        noise_model = NoiseModel()
        p1 = 0.001  # Single-qubit gate error rate
        p2 = 0.01   # CNOT gate error rate
        
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['u1', 'u2', 'u3', 'h', 'x'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx', 'mcx'])
    
    # Select simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    for k in k_values:
        for n in n_values:
            if n > k:
                continue
                
            i0 = generate_i0(k)
            l = math.ceil(2*k / n)
            param_start_time = time.time()  # Individual parameter set timing start
            
            try:
                # Build circuit
                qc = build_quantum_algorithm(k, n, l, i0)
                
                # Circuit analysis
                gate_counts = qc.count_ops()
                total_gates = sum(gate_counts.values())
                total_qubits = qc.num_qubits
                depth = qc.depth()
                
                # Simulation execution
                exec_start = time.time()
                result = execute(qc, simulator, shots=shots, noise_model=noise_model).result()
                exec_time = time.time() - exec_start  # Simulation time
                
                # Calculate success probability
                counts = result.get_counts()
                i0_binary = format(i0, f'0{k}b')
                success_counts = counts.get(i0_binary, 0)
                success_prob = success_counts / shots
                
                # Total time for this parameter set
                param_total_time = time.time() - param_start_time
                
                # Store key results
                results.append({
                    'k': k, 
                    'n': n, 
                    'total_gates': total_gates,
                    'simulation_time': exec_time,
                    'total_qubits': total_qubits,
                    'success_prob': success_prob
                })
                
                print(f"Completed: k={k}, n={n} | Success probability: {success_prob:.4f} | Total gates: {total_gates} | Total qubits: {total_qubits}")
                
            except Exception as e:
                print(f"Error: k={k}, n={n} - {str(e)}")
                continue
    
    # Total time statistics
    total_time = time.time() - total_start_time
    print(f"\nTotal runtime for all parameter combinations: {total_time:.2f} seconds")
    
    return results, total_time

# 5. Visualize results (only generate three key images)
def visualize_results(results):
    """Visualize key metrics as functions of k"""
    if not results:
        print("No results to visualize")
        return
        
    # Extract unique k and n values
    k_values = sorted(list(set(r['k'] for r in results)))
    n_values = sorted(list(set(r['n'] for r in results)))
    
    # 1. Number of quantum gates vs k
    plt.figure(figsize=(10, 6))
    for n in n_values:
        gate_counts = []
        valid_k = []
        
        for k in k_values:
            for r in results:
                if r['k'] == k and r['n'] == n:
                    gate_counts.append(r['total_gates'])
                    valid_k.append(k)
                    break
        
        plt.plot(valid_k, gate_counts, 'o-', label=f'n={n}')
    
    plt.xlabel('k (Search space size = 2^k)')
    plt.ylabel('Number of quantum gates')
    plt.title('Number of quantum gates vs k')
    plt.semilogy()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'gates_vs_k.png'))
    plt.show()
    
    # 2. Number of qubits vs k
    plt.figure(figsize=(10, 6))
    for n in n_values:
        qubits = []
        valid_k = []
        
        for k in k_values:
            for r in results:
                if r['k'] == k and r['n'] == n:
                    qubits.append(r['total_qubits'])
                    valid_k.append(k)
                    break
        
        plt.plot(valid_k, qubits, 's-', label=f'n={n}')
    
    plt.xlabel('k (Search space size = 2^k)')
    plt.ylabel('Number of qubits')
    plt.title('Number of qubits vs k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'qubits_vs_k.png'))
    plt.show()
    
    # 3. Success probability vs k
    plt.figure(figsize=(10, 6))
    for n in n_values:
        probs = []
        valid_k = []
        
        for k in k_values:
            for r in results:
                if r['k'] == k and r['n'] == n:
                    probs.append(r['success_prob'])
                    valid_k.append(k)
                    break
        
        plt.plot(valid_k, probs, '^-', label=f'n={n}')
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50% threshold')
    plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.3, label='90% threshold')
    plt.xlabel('k (Search space size = 2^k)')
    plt.ylabel('Success probability')
    plt.title('Success probability vs k')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'success_prob_vs_k.png'))
    plt.show()

# 6. Main function to execute simulation and visualization
def main():
    # Test parameter settings
    k_values = [2, 3, 4, 5]  # Test k values
    n_values = [2, 3, 4]     # Test n values
    shots = 5000             # Number of shots
    
    print(f"Starting quantum algorithm simulation...")
    print(f"Parameters: k={k_values}, n={n_values}, number of shots={shots}")
    print(f"Images will be saved to: {FIGURE_DIR}")
    
    # Execute performance evaluation
    results, total_time = evaluate_performance(k_values, n_values, shots, noise=False)
    
    # Save results JSON file to image directory
    import json
    with open(os.path.join(FIGURE_DIR, 'quantum_statistics_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_results(results)
    
    # Print summary statistics
    print("\n===== Quantum Algorithm Key Metrics Summary =====")
    print(f"Total runtime: {total_time:.2f} seconds")
    print("-" * 100)
    print(f"{'k':<4} | {'n':<4} | {'Total gates':<10} | {'Simulation time(s)':<16} | {'Total qubits':<12} | {'Success prob':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['k']:<4} | {r['n']:<4} | {r['total_gates']:<10} | {r['simulation_time']:.6f}        | {r['total_qubits']:<12} | {r['success_prob']:.4f}")
    print("-" * 100)

if __name__ == "__main__":
    main()
    