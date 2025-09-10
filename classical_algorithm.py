import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import math
import json
import re

# Force font settings, prioritizing fonts with complete symbol support
def setup_font():
    # Font priority: prefer fonts with complete symbol support
    preferred_fonts = [
        "WenQuanYi Micro Hei",  # Good symbol support
        "Arial Unicode MS",     # Contains almost all Unicode symbols
        "Heiti TC",
        "SimHei",
        "Microsoft YaHei",
        "SimSun"
    ]
    
    # Get available system fonts
    available_fonts = set()
    for font_path in fm.findSystemFonts():
        try:
            font_name = fm.FontProperties(fname=font_path).get_name()
            available_fonts.add(font_name)
        except:
            continue
    
    # Select the first available preferred font
    for font in preferred_fonts:
        if font in available_fonts:
            print(f"Using font: {font} (supports complete symbol set)")
            matplotlib.rcParams["font.family"] = font
            return font
    
    # If no preferred fonts found, use default settings with warning
    print("Warning: Ideal fonts not found, display issues may persist")
    matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    return None

# Ultimate symbol replacement: replace all possible minus signs and problematic symbols
def sanitize_text(text):
    # Convert to string (prevent non-string types)
    text = str(text)
    
    # Replace various minus sign variants with regular minus sign
    text = re.sub(r'[\u2212\u002D\u00AD\u2010-\u2015\uFE63\uFF0D]', '-', text)
    
    # Replace other potentially problematic symbols
    text = text.replace('²', '^2').replace('³', '^3')  # Replace superscripts with more universal notation
    
    return text

# Ensure result save directory exists
FIGURE_DIR = r"D:\quantum\fig"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(os.path.join(FIGURE_DIR, "classical"), exist_ok=True)
os.makedirs(os.path.join(FIGURE_DIR, "comparison"), exist_ok=True)

# 1. Define problem parameters and auxiliary functions identical to quantum algorithm
def generate_i0(k):
    """Generate target index i0 based on k value (consistent with quantum algorithm)"""
    if k == 2:
        return 1  # Binary representation: 01
    else:  # k >= 3
        return 5 % (2**k)

def g(x, n):
    """Define g(x) function: x XOR 0^(n-2)10 (consistent with quantum algorithm)"""
    mask = (1 << (n-1))  # (n-1)th bit is 1, others are 0
    return x ^ mask

def f(i, y, n, i0):
    """Define f(i, y) function (consistent with quantum algorithm)"""
    if i == i0:
        return 1  # Constant
    else:
        i_trunc = i % (2**n)
        return i_trunc ^ y

# 2. Classical algorithm implementation
def classical_algorithm(k, n, i0, l, trials=1):
    """Classical algorithm implementation"""
    total_operations = 0
    total_time = 0
    success_count = 0
    
    np.random.seed(42)
    x_list = [np.random.randint(0, 2**n) for _ in range(l+1)]
    g_list = [g(x, n) for x in x_list]
    total_operations += 2 * (l + 1)
    
    for _ in range(trials):
        start_time = time.time()
        
        for i in range(2**k):
            total_operations += 1
            
            valid = True
            for j in range(l):
                total_operations += 1
                
                f_j = f(i, g_list[j], n, i0)
                f_j1 = f(i, g_list[j+1], n, i0)
                total_operations += 2
                
                if f_j != f_j1:
                    valid = False
                    break
                total_operations += 1
            
            if valid:
                if i == i0:
                    success_count += 1
                break
        
        trial_time = time.time() - start_time
        total_time += trial_time
    
    success_prob = success_count / trials
    avg_time = total_time / trials
    
    return success_prob, total_operations, avg_time

# 3. Classical algorithm performance evaluation
def evaluate_classical_performance(k_values, n_values, trials=100):
    """Performance evaluation"""
    results = []
    total_start_time = time.time()
    save_dir = os.path.join(FIGURE_DIR, "classical")
    
    for k in k_values:
        for n in n_values:
            if n > k:
                continue
                
            i0 = generate_i0(k)
            l = math.ceil(2*k / n)
            
            try:
                success_prob, operations, avg_time = classical_algorithm(
                    k, n, i0, l, trials=trials
                )
                
                search_space = 2**k
                
                results.append({
                    'k': k,
                    'n': n,
                    'search_space': search_space,
                    'total_operations': operations,
                    'success_prob': success_prob,
                    'average_time': avg_time,
                    'trials': trials,
                    'l': l
                })
                
                print(f"Completed: k={k}, n={n} | Success probability: {success_prob:.4f} | Total operations: {operations:,} | Average time: {avg_time:.6f}s")
                
            except Exception as e:
                print(f"Error: k={k}, n={n} - {str(e)}")
                continue
    
    total_time = time.time() - total_start_time
    print(f"\nTotal runtime for all parameter combinations: {total_time:.2f}s")
    
    with open(os.path.join(save_dir, 'classical_statistics_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, total_time

# 4. Classical vs quantum algorithm comparison visualization (fully fix display issues)
def visualize_classical_quantum_comparison(classical_results, quantum_results):
    """Visualization function, completely solving text display issues"""
    if not classical_results or not quantum_results:
        print("Insufficient results for comparison")
        return
        
    # Set up font
    used_font = setup_font()
    
    # Extract unique k and n values
    k_values = sorted(list(set(r['k'] for r in classical_results)))
    n_values = sorted(list(set(r['n'] for r in classical_results)))
    comparison_dir = os.path.join(FIGURE_DIR, "comparison")
    
    # 1. Operations/gates comparison
    plt.figure(figsize=(12, 7))
    for n in n_values:
        classical_ops = []
        quantum_gates = []
        valid_k = []
        
        for k in k_values:
            c_result = next((r for r in classical_results 
                           if r['k'] == k and r['n'] == n), None)
            q_result = next((r for r in quantum_results 
                           if r['k'] == k and r['n'] == n), None)
            
            if c_result and q_result:
                classical_ops.append(c_result['total_operations'])
                quantum_gates.append(q_result['total_gates'])
                valid_k.append(k)
        
        # All text undergoes symbol cleaning
        plt.plot(valid_k, classical_ops, 'o-', 
                label=sanitize_text(f'n={n} (Classical operations)'))
        plt.plot(valid_k, quantum_gates, 's--', 
                label=sanitize_text(f'n={n} (Quantum gates)'))
    
    plt.xlabel(sanitize_text('k (Search space size = 2^k)'))
    plt.ylabel(sanitize_text('Number of operations/gates'))
    plt.title(sanitize_text('Comparison of Classical Operations vs Quantum Gates'))
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'operations_vs_gates.png'), dpi=300)
    plt.close()
    
    # 2. Runtime comparison
    plt.figure(figsize=(12, 7))
    for n in n_values:
        classical_times = []
        quantum_times = []
        valid_k = []
        
        for k in k_values:
            c_result = next((r for r in classical_results 
                           if r['k'] == k and r['n'] == n), None)
            q_result = next((r for r in quantum_results 
                           if r['k'] == k and r['n'] == n), None)
            
            if c_result and q_result:
                classical_times.append(c_result['average_time'])
                quantum_times.append(q_result['simulation_time'])
                valid_k.append(k)
        
        plt.plot(valid_k, classical_times, 'o-', 
                label=sanitize_text(f'n={n} (Classical algorithm)'))
        plt.plot(valid_k, quantum_times, 's--', 
                label=sanitize_text(f'n={n} (Quantum algorithm)'))
    
    plt.xlabel(sanitize_text('k (Search space size = 2^k)'))
    plt.ylabel(sanitize_text('Average runtime (seconds)'))
    plt.title(sanitize_text('Classical vs Quantum Algorithm Runtime Comparison'))
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'time_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Success probability comparison
    plt.figure(figsize=(12, 7))
    for n in n_values:
        classical_probs = []
        quantum_probs = []
        valid_k = []
        
        for k in k_values:
            c_result = next((r for r in classical_results 
                           if r['k'] == k and r['n'] == n), None)
            q_result = next((r for r in quantum_results 
                           if r['k'] == k and r['n'] == n), None)
            
            if c_result and q_result:
                classical_probs.append(c_result['success_prob'])
                quantum_probs.append(q_result['success_prob'])
                valid_k.append(k)
        
        plt.plot(valid_k, classical_probs, 'o-', 
                label=sanitize_text(f'n={n} (Classical algorithm)'))
        plt.plot(valid_k, quantum_probs, 's--', 
                label=sanitize_text(f'n={n} (Quantum algorithm)'))
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, 
               label=sanitize_text('100% success rate'))
    plt.xlabel(sanitize_text('k (Search space size = 2^k)'))
    plt.ylabel(sanitize_text('Success probability'))
    plt.title(sanitize_text('Classical vs Quantum Algorithm Success Probability'))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'success_prob_comparison.png'), dpi=300)
    plt.close()

# 5. Main function
def main():
    # Set up font in advance to ensure all text elements use correct font
    setup_font()
    
    k_values = [2, 3, 4, 5]
    n_values = [2, 3, 4]
    trials = 100
    
    print(f"Starting classical algorithm simulation...")
    print(f"Parameters: k={k_values}, n={n_values}, number of trials={trials}")
    print(f"Results will be saved to: {FIGURE_DIR}")
    
    classical_results, classical_total_time = evaluate_classical_performance(
        k_values, n_values, trials=trials
    )
    
    try:
        quantum_results_path = os.path.join(
            FIGURE_DIR, "no_noise", "quantum_statistics_results_no_noise.json"
        )
        with open(quantum_results_path, 'r') as f:
            quantum_results = json.load(f)
        print("\nQuantum algorithm results loaded, preparing comparison charts...")
        visualize_classical_quantum_comparison(classical_results, quantum_results)
    except FileNotFoundError:
        print("\nQuantum algorithm results file not found, cannot generate comparison charts")
        print("Please run the quantum algorithm code first to generate the results file")
    except Exception as e:
        print(f"Error loading quantum algorithm results: {str(e)}")
    
    print("\n===== Classical Algorithm Key Metrics Summary =====")
    print(f"Total runtime: {classical_total_time:.2f}seconds")
    print("-" * 110)
    print(f"{'k':<4} | {'n':<4} | {'Search space':<10} | {'Total ops':<12} | {'Success prob':<12} | {'Avg time(s)':<14}")
    print("-" * 110)
    for r in classical_results:
        print(f"{r['k']:<4} | {r['n']:<4} | {r['search_space']:<10} | {r['total_operations']:<12,} | {r['success_prob']:.4f}        | {r['average_time']:.6f}")
    print("-" * 110)

if __name__ == "__main__":
    main()
    