import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt.solvers import qp


#-------Exemplary 1-qubit and 2-qubit unitaries for the embedding circuit-------#
# Define the block of gates as a function
def V_block(x, wire):
    qml.RY(x, wires=wire)
    qml.RZ(x, wires=wire)

# Define the U(theta) block as a function
def U_block(theta, wires):
    # theta is a list or array of four parameters
    theta1, theta2, theta3, theta4 = theta
    
    # Apply operations on the first qubit (wire 0)
    qml.RY(theta1, wires=wires[0])
    qml.RZ(theta3, wires=wires[0])
    
    # Apply operations on the second qubit (wire 1)
    qml.RY(theta2, wires=wires[1])
    qml.RZ(theta4, wires=wires[1])
    
    # Apply CNOT (CX) with wire 0 as control and wire 1 as target
    qml.CNOT(wires=wires)


#-------Light-Cone Feature Selection class-------#
class LC_FS:
    def __init__(self, L, n, V_block_fn=V_block, U_block_fn=U_block):
            """
            :param L: Number of layers in the quantum circuit.
            :param n: Number of qubits (features) to consider.
            :param V_block_fn: Function for single-qubit gate block (default is V_block).
            :param U_block_fn: Function for two-qubit gate block (default is U_block).
            """
            self.L = L
            self.n = n
            self.wires = list(range(n))
            self.dev = qml.device('default.qubit', wires=n)
            self.Lambda = np.ones((n,)) / n  # Initial lambda is uniform
            self.partial_RHOS = None

            # Assign the default or user-defined block functions
            self.V_block = V_block_fn
            self.U_block = U_block_fn

            self.thetas = np.random.random((L,(n//2),4), requires_grad=True)

    #-------- Main methods --------

    def LC_importance_score(self,X, Y, optimize=True, episodes=3, max_iterations=500, conv_tol=1.e-4, stepsize = 0.5):
        """
        The high-level function which gathers all the necessary steps and returns importance scores for features
        """
        self.get_partial_traces(X,self.thetas)
        kernels = self.base_kernels_matrices()
        centered_kernels = self.center_base_kernels_matrices(kernels)
        if(optimize):
            self.optimize_parameters(X,Y,episodes=episodes, max_iterations=max_iterations, conv_tol=conv_tol, stepsize=stepsize)
        Ps = self.importance_score()
        return(Ps)
    
    def LC_feature_selection(self, X, Y, optimize=True, episodes=3, max_iterations=500, conv_tol=1.e-4, stepsize = 0.5):
        """
        The high-level function which gathers all the necessary steps and returns importance scores for features and the list of features sorted by their importance
        """
        # Get importance scores
        Ps = self.LC_importance_score(X,Y,optimize=optimize, episodes=episodes, max_iterations=max_iterations, conv_tol=conv_tol, stepsize=stepsize)
        # Combine the list and vector using zip, then sort by the vector values in descending order
        combined = sorted(zip(self.wires, Ps), key=lambda pair: pair[1], reverse=True)
        
        # Unzip the sorted pairs back into two lists
        sorted_wires, sorted_Ps = zip(*combined)
        sorted_Ps = [float(p) for p in list(sorted_Ps)]
        # Convert them back to list form
        return list(sorted_wires), sorted_Ps
        
    def optimize_thetas(self, X, Y, thetas, num_steps=100, learning_rate=0.01):
            """
            Optimizes the `thetas` variational parameters using a gradient-based optimizer (Adam) 
            to maximize the centered alignment.
            
            :param X: Input data (features).
            :param Y: Labels.
            :param thetas: Initial variational parameters for the quantum circuit.
            :param num_steps: Number of optimization steps.
            :param learning_rate: Learning rate for the optimizer.
            :return: Optimized `thetas`.
            """
            opt = qml.AdamOptimizer(stepsize=learning_rate)

            # Objective function to maximize centered alignment
            def objective(thetas):
                self.get_partial_traces(X, thetas)
                alignment = self.centered_alignment(Y)
                return -alignment  # Negative because we want to maximize the alignment

            thetas_opt = thetas.copy()

            print(objective(thetas_opt))

            for i in range(num_steps):
                thetas_opt = opt.step(objective, thetas_opt)
                if (i + 1) % 10 == 0:
                    current_alignment = -objective(thetas_opt)
                    print(f"Step {i+1}: Centered alignment = {current_alignment}")

            return thetas_opt

    #-------- Quantum computation tools --------

    def LC_embedding_map(self, x, thetas):
        """
            The embedding circuit
        """

        self.thetas = thetas
        N = len(self.wires)
        shifted = False
        for l in range(self.L):
        
            qml.Barrier() # Barriers are drawn for more convenient circuit drawing
            #single-qubit gates
            for wire in range(N):
                V_block(x[wire],wire=wire)
            
            qml.Barrier()
            #two-qubit gates
            if(shifted):
                wires_2=[[id,id+1] for id in list(range(N)) if id%2==1]
                wires_2[-1][-1]=0
            else:
                wires_2=[[id,id+1] for id in list(range(N)) if id%2==0]
            for i,wire in enumerate(wires_2):
                U_block(thetas[l,i],wires=wire)
            shifted = not shifted # Every second layer we shift the brick structure of the 2-qubit gates

    def density_matrix(self, x, thetas):
        """
        Function which introduces a proper qnode for pennylane, creates actual circuit and calculates the density matrix
        """
        # Define the qnode inside the method
        @qml.qnode(self.dev)
        def circuit():
            self.LC_embedding_map(x, thetas)
            return qml.density_matrix(wires=self.wires)
        
        # Execute and return the result of the qnode
        return circuit()
    
    def partial_traces(self,x, thetas):
        """
        Function which returns 1-qubit reduced density matrices for the circuit
        """
        rho = self.density_matrix(x, thetas)
        partial_rhos = [qml.math.partial_trace(rho, indices=w) for w in self.wires_to_trace_out()]
        return(np.stack(partial_rhos))

    def get_partial_traces(self,X, thetas, pass_rhos=False):
        """
        Calculates 1-qubit reduced density matrices for the circuit and stores it in the partial_RHOS variable
        """
        #partial_RHOS = np.empty((X.shape[0],X.shape[1],2,2), dtype=np.complex128)
        partial_RHOS = []
        for x in X:
            partial_RHOS.append(self.partial_traces(x,thetas))

        if(pass_rhos):
            return(np.stack(partial_RHOS))
        else:
            self.partial_RHOS = np.stack(partial_RHOS)
        


    def base_kernels_value(self, id1, id2):
        """
        Obtain the values of the 1-qubit components of the net kernel for the datapoints with indices id1, id2
        """
        #if(self.partial_RHOS.all() == None):
        #    print('Error! Partial traces not calculated yet, run .get_partial_traces(X,thetas) method.')
        #    return(-1)
        rho1 = self.partial_RHOS[id1]
        rho2 = self.partial_RHOS[id2]
        kernel_components = np.array(np.real([np.matrix.trace(np.matmul(rho1, rho2)[i]) for i in range(rho1.shape[0])]), dtype=float)
        return(kernel_components)

    def kernel_value(self,id1, id2):
        """
        Get the value of the net kernel between two datapoints with indices id1, id2
        """
        rho1 = self.partial_RHOS[id1]
        rho2 = self.partial_RHOS[id2]
        #kernel_components = [np.matrix.trace(np.matmul(rho1, rho2)[i]) for i in range(rho1.shape[0])]
        
        #kernel_components = [np.matrix.trace(np.matmul(rho1, rho2)[i]) for i in range(rho1.shape[0])]
        #kernel_components = np.stack(kernel_components)
        kernel_components = np.einsum('nij,nij->n', rho1, rho2)
        k = np.real(np.inner(np.ones((kernel_components.shape[0],))/kernel_components.shape[0],kernel_components))
        return(k)

    def get_kernel_value(self, partial_rhos, id1, id2):
        """
        Get the value of the net kernel between two datapoints with indices id1, id2
        """
        rho1 = partial_rhos[id1]
        rho2 = partial_rhos[id2]
        #kernel_components = [np.matrix.trace(np.matmul(rho1, rho2)[i]) for i in range(rho1.shape[0])]
        
        #kernel_components = [np.matrix.trace(np.matmul(rho1, rho2)[i]) for i in range(rho1.shape[0])]
        #kernel_components = np.stack(kernel_components)
        kernel_components = np.einsum('nij,nij->n', rho1, rho2)
        k = np.real(np.inner(np.ones((kernel_components.shape[0],))/kernel_components.shape[0],kernel_components))
        return(k)
    
    def kernel_matrix(self):
        """
        Get the full net kernel matrix
        """
        #if(self.partial_RHOS.all() == None):
        #    if(thetas==None):
        #        self.get_partial_traces(X, self.thetas)
        #    else:
        #        self.get_partial_traces(X, thetas)
        # Initialize the kernel matrix
        m = self.partial_RHOS.shape[0]
        K = np.zeros((m, m), dtype=type(self.kernel_value(0, 1)))
        
        # Populate the kernel matrix using the kernel_value method
        for i in range(m):
            for j in range(i+1, m):  # Take advantage of symmetry (K is symmetric)
                K[i, j] = self.kernel_value(i, j)
                K[j, i] = K[i, j]  # Symmetric entry
        K = K + np.identity(m)

        return K

    def get_kernel_matrix(self):
        """
        Get the full net kernel matrix
        """
        #if(self.partial_RHOS.all() == None):
        #    if(thetas==None):
        #        self.get_partial_traces(X, self.thetas)
        #    else:
        #        self.get_partial_traces(X, thetas)
        # Initialize the kernel matrix
        m = self.partial_RHOS.shape[0]
        K = np.zeros((m, m), dtype=type(self.kernel_value(0, 1)))
        
        # Populate the kernel matrix using the kernel_value method
        for i in range(m):
            for j in range(i+1, m):  # Take advantage of symmetry (K is symmetric)
                K[i, j] = self.kernel_value(i, j)
                K[j, i] = K[i, j]  # Symmetric entry
        K = K + np.identity(m)
        K = np.array([[x._value if isinstance(x, np.numpy_boxes.ArrayBox) else x for x in row] for row in K])
        return K

    def get_train_kernel_matrix(self, X):
        """
        Get the kernel matrix for training on the data X
        """
        #if(self.partial_RHOS.all() == None):
        #    if(thetas==None):
        #        self.get_partial_traces(X, self.thetas)
        #    else:
        #        self.get_partial_traces(X, thetas)
        # Initialize the kernel matrix
        partial_rhos = self.get_partial_traces(X, self.thetas, pass_rhos=True)
        m = partial_rhos.shape[0]
        K = np.zeros((m, m), dtype=type(self.get_kernel_value(partial_rhos, 0, 1)))
        
        # Populate the kernel matrix using the kernel_value method
        for i in range(m):
            for j in range(i+1, m):  # Take advantage of symmetry (K is symmetric)
                K[i, j] = self.get_kernel_value(partial_rhos, i, j)
                K[j, i] = K[i, j]  # Symmetric entry
        K = K + np.identity(m)
        K = np.array([[x._value if isinstance(x, np.numpy_boxes.ArrayBox) else x for x in row] for row in K])
        return K

    def get_test_kernel_matrix(self, X_SV, X_test):
        """
        Get the kernel matrix for test on the data X_SV, X_test.
        For kernel=”precomputed”, the expected shape of X is (n_samples_test, n_samples_train).
        """
        partial_rhos_SV = self.get_partial_traces(X_SV, self.thetas, pass_rhos=True)
        partial_rhos_test = self.get_partial_traces(X_test, self.thetas, pass_rhos=True)
        partial_rhos = np.vstack([partial_rhos_test, partial_rhos_SV])

        m_SV = partial_rhos_SV.shape[0]
        m_test = partial_rhos_test.shape[0]

        K = np.zeros((m_test, m_SV), dtype=type(self.get_kernel_value(partial_rhos_SV, 0, 1)))
        
        # Populate the kernel matrix using the kernel_value method
        for i_test in range(m_test):
            for i_SV in range(m_SV):
                K[i_test, i_SV] = self.get_kernel_value(partial_rhos, i_test, m_test+i_SV)
        return K

    def base_kernels_matrices(self):
        """
        Obtain the matrices of the 1-qubit components of the net kernel
        """
        m = self.partial_RHOS.shape[0]
        kernels = np.zeros((self.n,m,m))

        # Populate the kernel matrix using the kernel_value method
        for i in range(m):
            for j in range(i+1, m):  # Take advantage of symmetry (K is symmetric)
                kernels[:,i, j] = self.base_kernels_value(i, j)
                kernels[:,j, i] = kernels[:,i, j]  # Symmetric entry
        kernels = np.array([k+np.identity(m) for k in kernels])

        return kernels
    
    def center_base_kernels_matrices(self, kernels):
        """
        Take matrices of the components of the net kernel matrix and center them
        """
        kernels_centered = np.array([self.center_matrix(k) for k in kernels])
        return(kernels_centered)
    
    def kernel_target_alignment(self, labels):
        """
        Compute a simple Kernel-Target alignment defined in: Cristianini, Nello, et al. "On kernel-target alignment." Advances in neural information processing systems 14 (2001).
        """
        # Compute the kernel matrix using the compute_kernel_matrix method
        K = self.kernel_matrix()
        
        # Create the ideal kernel matrix T (outer product of labels)
        T = np.outer(labels, labels)  # Ideal kernel (label outer product)
        
        # Calculate the Frobenius inner product <K, T> and norms
        frobenius_inner_product = np.sum(K * T)
        norm_K = np.linalg.norm(K, 'fro')
        norm_T = np.linalg.norm(T, 'fro')

        # Kernel target alignment
        kta = frobenius_inner_product / (norm_K * norm_T)
        
        return kta

    def centered_alignment(self,Y):
        """
        Compute a simple Kernel-Target alignment defined in: Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh. "Algorithms for learning kernels based on centered alignment." The Journal of Machine Learning Research 13 (2012): 795-828.
        """
        # Step 1: Compute the kernel matrix K
        K = self.kernel_matrix()
        
        # Step 2: Center the kernel matrix K
        K_centered = self.center_matrix(K)
        
        # Step 3: Create the ideal kernel matrix T (outer product of labels)
        T = np.outer(Y, Y)
        
        # Step 4: Center the target matrix T
        T_centered = self.center_matrix(T)
        
        # Compute frobenius products by hand, as numpy implementation does not work well with autograd
        product_KT = K_centered * T_centered
        frobenius_inner_product = 0
        for i in range(product_KT.shape[0]):
            for j in range(product_KT.shape[1]):
                  frobenius_inner_product += product_KT[i,j]

        product_KK = K_centered * K_centered
        norm_K_centered = 0
        for i in range(product_KK.shape[0]):
            for j in range(product_KK.shape[1]):
                  norm_K_centered += product_KK[i,j]
        norm_K_centered = np.sqrt(norm_K_centered)

        product_TT = T_centered * T_centered
        norm_T_centered = 0
        for i in range(product_TT.shape[0]):
            for j in range(product_TT.shape[1]):
                  norm_T_centered += product_TT[i,j]
        norm_T_centered = np.sqrt(norm_T_centered)
        
        # Step 6: Centered alignment
        centered_alignment_value = frobenius_inner_product / (norm_K_centered * norm_T_centered)
        
        return centered_alignment_value
    
    def optimize_lambda(self, kernels, Y):
        """
        Optimize the value of lambda according to the "alignf" quadratic programming routine introduced in: Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh. "Algorithms for learning kernels based on centered alignment." The Journal of Machine Learning Research 13 (2012): 795-828.
        Utilizes external cvxopt library.
        """
        print("-------------------")
        print("Lambdas optimization")
        #Prepare optimized function
        p = kernels.shape[0]
        T = np.outer(Y, Y)
        
        M = np.zeros((p,p))
        for i in range(p):
            for j in range(p):
                M[i,j] = self.matrix_frobenius_product(kernels[i], kernels[j])

        a = np.array([self.matrix_frobenius_product(k,T) for k in kernels])
        
        G = matrix(0.0, (p,p))
        G[::p+1] = -1.0
        h = matrix(0.0, (p,1))
        A = matrix(0.0, (1,p))
        b = matrix(0.0)
        P = matrix(M)
        q = -matrix(a)

        v = qp(P, q, G, h)['x']
        v_a = np.array([vv for vv in v])
        Lambda_opt = v_a/np.linalg.norm(v_a, ord=1)
        self.Lambda = Lambda_opt

    def objective_function(self, X, Y, thetas):
        self.get_partial_traces(X, thetas)
        K = self.kernel_matrix()
        r = -self.centered_alignment(Y)
        return(r)

    def optimize_thetas(self, X, Y, max_iterations=500, conv_tol = 1e-04, stepsize=0.5):
        print("-------------------")
        print("Thetas optimization")
        opt = qml.GradientDescentOptimizer(stepsize=stepsize)
        cost = [self.objective_function(X,Y, self.thetas)]
        
        for step in range(max_iterations):
            self.thetas, prev_cost = opt.step_and_cost(lambda theta: self.objective_function(X,Y,theta), self.thetas)

            if step % 10 == 0:
                print(f"Step = {step},  Cost function = {cost[-1]:.8f} ")
            
            if(np.abs(cost[-1] - prev_cost) <= conv_tol and step>0):
                break
            cost.append(prev_cost)
        
        print("-------------------")
        return(cost)
    
    def optimize_parameters(self, X, Y, episodes=3, max_iterations=500, conv_tol = 1e-04, stepsize=0.5):
        cost = []
        for _ in range(episodes):
            self.get_partial_traces(X,self.thetas)
            kernels = self.base_kernels_matrices()
            centered_kernels = self.center_base_kernels_matrices(kernels)
            self.optimize_lambda(centered_kernels,Y)

            cost_episode = self.optimize_thetas(X,Y,max_iterations=max_iterations, conv_tol=conv_tol, stepsize=stepsize)
            cost.append(cost_episode)
        
        return(cost)
    
    def light_cone_weigths(self,lst):
        """
        Tool to compute light cone weigths w_{lambda_l} from the paper: Suzuki, Yudai, Rei Sakuma, and Hideaki Kawaguchi. "Light-cone feature selection for quantum machine learning." arXiv preprint arXiv:2403.18733 (2024).
        The input list contains everywhere zeros except in the pair of wires which are connected to the measurement in the last layer. Their value is L, the number of layers.
        """
        n = len(lst)
        
        # Find the index of the leftmost L
        left_L_index = lst.index(self.L)

        # Create first list (going to the left)
        first_list = [0] * n
        first_list[left_L_index] = self.L
        first_list[left_L_index + 1] = self.L

        # Fill to the left, wrapping around to the end
        value = self.L - 1
        i = (left_L_index - 1) % n
        while i != left_L_index:
            first_list[i] = value
            value -= 1
            i = (i - 1) % n

        # Create second list (going to the right)
        second_list = [0] * n
        second_list[left_L_index] = self.L
        second_list[left_L_index + 1] = self.L

        # Fill to the right, wrapping around to the start
        value = self.L - 1
        i = (left_L_index + 2) % n
        while i != left_L_index:
            second_list[i] = value
            value -= 1
            i = (i + 1) % n

        # Apply element-wise max and reset negative values to 0
        result = [max(first_list[i], second_list[i]) for i in range(len(first_list))]
        
        # Reset negative values to 0
        result = [max(0, value) for value in result]

        return result

    def importance_weights(self, wire):
        """
        Compute light cone weigths w_{lambda_l} for the given wire. Based on: Suzuki, Yudai, Rei Sakuma, and Hideaki Kawaguchi. "Light-cone feature selection for quantum machine learning." arXiv preprint arXiv:2403.18733 (2024).
        """
        w = [0]*self.n
        shifted = not (self.L % 2)
        if(shifted):
            wires_2=[[id,id+1] for id in list(range(self.n)) if id%2==1]
            wires_2[-1][-1]=0
        else:
            wires_2=[[id,id+1] for id in list(range(self.n)) if id%2==0]
        
        pair_index = self.find_pair_index(wires_2, wire)
        multiplicities = wires_2[pair_index]
        for index in multiplicities:
            w[index] = self.L
        return(self.light_cone_weigths(w))
    
    def importance_score(self):
        """
        Return importance score, defined in: Suzuki, Yudai, Rei Sakuma, and Hideaki Kawaguchi. "Light-cone feature selection for quantum machine learning." arXiv preprint arXiv:2403.18733 (2024).
        """

        w_matrix = np.zeros((self.n,self.n))
        for w in range(w_matrix.shape[0]):
            w_matrix[w] = self.importance_weights(w)
        
        Ps = []
        for wire in self.wires:
            w = w_matrix[:,wire]
            Ps.append(np.dot(w,self.Lambda))
        Ps = np.array(Ps)
        # Normalize
        Ps = Ps/Ps.sum()
        return(Ps)

    #-------- General tools --------

    def draw(self, x, thetas, style='black_white'):
        """
        Draw an embedding map circuit
        """
        # Define the qnode inside the method for drawing purposes
        @qml.qnode(self.dev)
        def circuit():
            self.LC_embedding_map(x, thetas)
            return qml.density_matrix(wires=self.wires)

        # Use the Pennylane sketch style
        qml.drawer.use_style(style)
        # Draw the circuit with matplotlib
        fig, ax = qml.draw_mpl(circuit)()

        # Show the plot
        plt.show()

    def wires_to_trace_out(self):
        """
        Returns n-copies of the wires list, each with single element removed.
        Used to indicate which wires to trace out to obtain 1-qubit reduced density matrices.
        """
        lst = self.wires
        result = []
        for i in range(len(lst)):
            new_lst = lst[:i] + lst[i+1:]  # Remove the i-th element
            result.append(new_lst)
        return result

    def center_matrix(self, M):
        """
        Center a matrix M
        """
        n = M.shape[0]
        # Center the matrix according to the formula
        one_n = np.ones((n, n)) / n
        M_centered = M - one_n @ M - M @ one_n + one_n @ M @ one_n
        return M_centered

    def matrix_frobenius_product(self,A,B):
        """
        Frobenius product between matrices A, B
        """
        return(np.trace(np.matmul(A.transpose(),B)))


    def find_pair_index(self, pairs, value):
        """
        Take the list of lists which contain pairs of numbers. A single number is always only in one of the pairs.
        Return the index of the pair in which the number is.
        Used for analysing the light-cone structure of the circuit.
        """
        for i, pair in enumerate(pairs):
            if value in pair:
                return i
        return -1  # Return -1 if the value is not found