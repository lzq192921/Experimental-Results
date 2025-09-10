Here, we provide comprehensive simulation results of Algorithm 2 (Paper: Conditional Constant Function Problem and Its Quantum Solutions:  Attacking  Feistel  Ciphers). 


For an $n$-bit input $x \in \{0,1\}^{n}$, $g(x) = x \oplus 0^{n-2}10$, where $(0^{n-2}10)_2$ is an $n$-bit binary number with the $(n-1)$-th bit set to 1 (all other bits 0).

\item Quantum Oracle $O_f$ for $f(i,y)$. The oracle $O_f|i\rangle|y\rangle|0\rangle = |i\rangle|y\rangle|f(i,y)\rangle$ encodes the problem's key property (only $i=i_0$ produces constant output, $i \in \{0,1\}^{k}$):


\begin{itemize}[leftmargin=*]
    \item If $i = i_0$ (unique target index): $f(i,y) = 0^{n-2}01$ (constant)

    
    \item If $i \neq i_0$: $f(i,y) = i_{\text{trunc}} \oplus y$, where $i_{\text{trunc}} = i \mod 2^{n}$.

    
    We present key results comparing \cref{alg:Simulator} with classical brute-force search across three noise configurations.
    Experimental Setup
    \begin{itemize}
    \item \textbf{Problem scales and simulation limits}: Search space sizes range from $2^2=4$ to $2^4=16$ ($k \in \{2,3,4\}$) with block sizes $n \in \{2,3,4\}$.

    
    \item Unique target index $i_0 = 5 \bmod 2^k$ (adjusted to fit $k$-bit binary representation: $i_0=1$ for $k=2$, $i_0=5$ for $k \geq 3$);

    
    \item \textbf{Noise configurations} (depolarizing models):
    \begin{itemize}[leftmargin=*]
        \item \textbf{No-noise}: Ideal (zero errors);
        
        \item \textbf{Low-noise}: State-of-the-art hardware: single-qubit 0.01\%, two-qubit 0.1\%;
        
        \item \textbf{Medium-noise}: Laboratory-grade hardware: single-qubit 0.1\%, two-qubit 1.0\%.
    \end{itemize}
    \item 5,000 independent shots were performed using Qiskit 0.43.3 (for quantum simulations) and Python 3.7.13 (for classical benchmarks);
    
    \item \textbf{Classical benchmark}: Brute-force search ($O(2^k)$ complexity).
\end{itemize}
