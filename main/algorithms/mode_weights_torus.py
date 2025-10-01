import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy import optimize # For the bandwidth selectors 'cv_ml' and 'cv_ls', which are not yet implemented.
from statsmodels.nonparametric import _kernel_base

########========--------- Points mean and variance on torus --------========########

def var_on_torus(X):
    n, d = X.shape
    scalar_product_with_ones = np.sum(X, axis=1)
    real, im = np.cos(2 * np.pi * scalar_product_with_ones), np.sin(2 * np.pi * scalar_product_with_ones)
    complex_mean = np.sum(real, axis=0)/n + 1j * np.sum(im, axis=0)/n
    complex_mean_module = np.abs(complex_mean)
    return -np.log(complex_mean_module)/(2*d*np.pi**2)

def mean_on_torus(X):
    n, d = X.shape
    real, im = np.cos(2 * np.pi * X), np.sin(2 * np.pi * X)
    complex_mean = np.sum(real, axis=0)/n + 1j * np.sum(im, axis=0)/n
    mean = (np.angle(complex_mean) + np.pi) / (2*np.pi)
    return mean

########========--------- Bandwidth selection methods --------========########

def get_bandwidth(X, bw, n_periods=3):
    # Wrapper for the different bandwidth selectors
    type_bw = type(bw)
    bw_dictionary = {'silverman':compute_bandwidth_silverman} #, 'cv_ml':compute_bandwidth_cv_ml, 'cv_ls':compute_bandwidth_cv_ls}
    if type_bw == float:
        return bw
    else:
        if bw not in bw_dictionary.keys():
            raise ValueError('The bandwidth bw given is not among the accepted values')
        return bw_dictionary[bw](X, n_periods)
    
def generate_grid(n, d):#Generate (n+1)^d grid points spanning [0,1]^d. Returns: np.ndarray: array of shape ((n+1)^d, d) containing the grid points
    lin = np.linspace(0, 1, n + 1)
    mesh = np.meshgrid(*([lin] * d), indexing="ij")
    points = np.stack([m.flatten() for m in mesh], axis=-1)
    return points
    
def compute_bandwidth_silverman(X, n_periods=3):
    n, d = X.shape

    n_quad_per_line = 50
    X_quad = generate_grid(n_quad_per_line, d)
    sq_std_dev = var_on_torus(X)

    integral_quantity = np.sum(integrand(X_quad, X, sq_std_dev, n_periods))/len(X_quad)

    return n**(-1/(d+4)) * (d/((4*np.pi)**(d/2) * integral_quantity) )**(1/(d+4))

def integrand(X_quad, X, sq_sigma, n_periods=3):
    summands = compute_summands(X_quad, X, sq_sigma, n_periods)
    return np.sum(summands, axis=1)**2

def compute_summands(X_quad, X, sq_sigma, n_periods=3):
    n, d = X.shape
    ranges = [np.arange(-n_periods, n_periods+1)] * d
    mesh = np.meshgrid(*ranges, indexing='ij')  # each mesh[i] shape: (2n_periods+1, ..., 2n_periods+1) (d times) (K=(2n_periods+1)**d)
    k_vals = np.stack(mesh, axis=-1).reshape(-1, d)  # shape ((2n_periods+1)^K, d)
    X_plus_k = X_quad[:, None, :] + k_vals[None, :, :]# shape (m, (2n_periods+1)^K, d)
    sq_distances = np.sum(X_plus_k ** 2, axis=2) #tableau n_quad, (2*n_periods+1)**d

    return compute_summand(sq_distances, sq_sigma, d)
    
def compute_summand(sq_distances, sq_sigma, d):
    return (sq_distances/sq_sigma + d) * rho(sq_distances, np.sqrt(sq_sigma), d)/sq_sigma



def compute_bandwidth_cv_ml(X, n_periods): # remains to be done... model: statsmodel.nonparametric
    n, d = X.shape
    return 

def compute_bandwidth_cv_ls(X, n_periods): # remains to be done... model: statsmodel.nonparametric
    n, d = X.shape
    return

########========--------- Projection on simplex --------========########

def projection_on_simplex(x):
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(x)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0.0)

########========--------- Gaussian computing methods --------========########

def rho(sq_distances, sigma, d): # computes the gaussian density
    return np.exp(-sq_distances/(2*sigma**2))/(sigma**d * (2*np.pi)**(d/2))

def compute_gaussian_density(X_query, sigma, n_periods=3): # Density of periodised Gaussian (limited to indexes $\{-n_periods, n_periods\}^d$) of mean 0 and variance sigma^2 I_d
    m, d = X_query.shape

    ranges = [np.arange(-n_periods, n_periods+1)] * d
    mesh = np.meshgrid(*ranges, indexing='ij')  # each mesh[i] shape: (2n_periods+1, ..., 2n_periods+1) (K times)
    k_vals = np.stack(mesh, axis=-1).reshape(-1, d)  # shape ((2n_periods+1)^K, d)

    X_plus_k = X_query[:, None, :] + k_vals[None, :, :]# shape (m, (2n_periods+1)^K, d)
    norms_squared = np.sum(X_plus_k ** 2, axis=2)
    rho_vals = rho(norms_squared, sigma, d)  # shape (mn, (2n_periods+1)^K)
    g_vals = np.sum(rho_vals, axis=1)  # shape (m,)# Sum over k to get g(x - x_i)
    return g_vals

def KDE_evaluation(X_train, X_eval, sigma, n_periods=3): # Returns a np.array $G$ of shape (len(X_1)), $G[i] \approx \frac{1}{len(X_2)} \sum_{j=1}^{len(X_2)} g_\sigma(X_1[i]-X_2[j])$ with $g$ the periodised Gaussian kernel
    m, d = X_eval.shape
    n = X_train.shape[0]
    
    ranges = [np.arange(-n_periods, n_periods+1)] * d
    mesh = np.meshgrid(*ranges, indexing='ij')  # each mesh[i] shape: (2n_periods+1, ..., 2n_periods+1) (K times)
    k_vals = np.stack(mesh, axis=-1).reshape(-1, d)  # shape ((2n_periods+1)^K, d)
    x_diff = X_eval[:, None, :] - X_train[None, :, :] # shape: (m, n, d)
    x_diff_flat = x_diff.reshape(-1, d) # Reshape to (m*n, d) so we can compute g on all at once
    x_plus_k = x_diff_flat[:, None, :] + k_vals[None, :, :]  # shape (mn, (2n_periods+1)^K, d) Compute x + k for each point in x_diff_flat and each k
    norms_squared = np.sum(x_plus_k ** 2, axis=2)  # shape (mn, (2n_periods+1)^K) # Compute squared norms ||x + k||^2

    rho_vals = rho(norms_squared, sigma, d)  # shape (mn, (2n_periods+1)^K)
    g_vals = np.sum(rho_vals, axis=1)  # shape (mn,)# Sum over k to get g(x - x_i)
    g_vals = g_vals.reshape(m, n)  # shape (m, n)# Reshape back to (m, n) and sum over i to get f(x)
    f_vals = np.sum(g_vals, axis=1)  # shape (m,)

    return f_vals/len(X_train)

########========--------- Gradient descent and estimator class --------========########

class GD_estimator:

    def __init__(self, KL_type: int, p0: np.ndarray, U: list, X: list, h: float, eps: float, max_iter: int, bw:list, n_periods:int=3):
        """
        Computes the KDE bandwidths and then the values of these KDE in the datapoints, which will be used at each step of the gradient descent. There is no output at this step. The bandwidths are kept in the attribute 'alpha'.

        Parameters
        ----------
        KL_type: integer, either 0 or 1
        If 0, the algorithm based on $\pi \mapsto KL(\mu, \pi)$ is used. If 1 it is $\pi \mapsto KL(\pi|\mu)$, \mu being the target and \pi the estimate.

        p0: np.ndarray of shape (K)
        Initial value of the weight vector for the gradient descent, must belong to the simplex $\Delta_{K-1}$.

        U: Sequence[np.ndarray] of length K
        Gibbs potential values of each mode. The values of mode $i$ are in $U[i]$ of shape $(n_i)$.

        X: Sequence[np.ndarray] of length K
        Datapoints of each mode. The data points of mode $i$ are in $X[i]$, a np.ndarray of shape $(n_i, d)$, corresponding to $U[i]$.

        h: float
        Gradient descent time step.

        eps: float
        A stopping criteria for the gradient descent. The descent ends when the euclidean distance between two consecutive weights is lower than 'eps' or if another stopping criteria got fulfilled earlier.

        max_iter: int
        A stopping criteria for the gradient descent. The descent ends when the number of time steps reaches max_iter or if another stopping criteria got fulfilled earlier.

        bw: Sequence[string, float]
        Bandwidth selection rules for the sampled densities KDE. The selection rule for mode $i$ is in $bw[i]$. The valid values are: a positive float (the bandwidth is fixed at this value); 'silverman' for the Silverman's rule of thumb which is appropriate for unimodal data and optimal in the AMISE sense for Gaussian data.

        n_periods: int
        Number of periods used to compute an approximation of the periodised Gaussian kernel on the unit torus. The set of idnexes used is $\{-n_periods, n_periods\}^d$. Default is 3

        Returns
        -------
        None (the computed bandwidth parameters are stored in the attribute alpha)
        """
        self.n_periods = n_periods # Number of periods in one direction considered for the density calculus. The space considered will be [-n_periods, 1 + n_periods]^d
        self.d = len(X[0][0]) # Space dimension
        self.K = len(X) # Number of clusters
        self.n = [0]*self.K # Number of data in each cluster, filled later in a for loop
        self.alpha = [0]*self.K # Bandwidth storage for each mode, filled later in a for loop
        self.h = h
        self.U = U
        self.X = X
        self.max_iter = max_iter
        self.eps = eps 
        self.stop_message = ""
        self.bw = bw
        self.KL_type = KL_type

        print('Starting bandwidth computations and KDE evaluations in the training points...')
        self.start_timer()
        self.KDE_evaluations_on_training = {} # Stores the values of $\hat{\nu}_i(x_{q,j})$ for $i,j,q \in \{1,\ldots,K\}$
        for k in range(self.K):
            self.n[k] = len(X[k])
            self.alpha[k] = get_bandwidth(X[k], bw[k], self.n_periods) # Computes the bandwidth for mode $k$ data.
            for q in range(self.K):
                self.KDE_evaluations_on_training[q + k*self.K] = KDE_evaluation(self.X[k], self.X[q], self.alpha[k], self.n_periods) # Computes \hat{\nu}_k(x_{.,q})
        self.stop_timer()
        print("Done. Took "+str(round(self.stop_time - self.start_time,2))+"s.")

        self.constant_part = [] # list of $e^{-U(x_{j,i})}/\hat{\nu}_{\alpha_i}(x_{j,i})$ (when KL_type=0), this is a constant coefficient along the descent
        
        for k in range(self.K):
            hat_nu_k = self.get_KDE_evaluations_on_training(k,k)

            if self.KL_type==0: self.constant_part.append( - np.exp(-U[k]) / hat_nu_k )
            else: self.constant_part.append( 1 / hat_nu_k )

        self.p_list = [p0]
        return
    
    def get_KDE_evaluations_on_training(self, i, j): # returns \hat{\nu}_i(x_{.,j})
        return self.KDE_evaluations_on_training[j + i*self.K]

    ####====---- Gradient descent functions ----====####

    def optimise(self):
        """
        Runs the gradient descent on the surrogate function of $\Delta_{K-1}$

        Parameters
        ----------
        None

        Returns
        -------
        None (the successive weights computed are available in the attribute p_list which is a list of np.ndarray of shape (K))
        """
        s=0
        self.start_timer()
        while self.check_stop(s):
            self.descent_step()
            s+=1
        self.stop_timer()
        print("Gradient descent done ; " + self.stop_message)
        print("Took "+str(round(self.stop_time - self.start_time,2))+"s.")
        return
    
    def start_timer(self):
        self.start_time = timer()
        return
    
    def stop_timer(self):
        self.stop_time = timer()
    
    def descent_step(self):
        old_p = self.p_list[-1]
        new_p =  old_p - self.h * self.gradient_p(old_p)
        self.p_list.append(projection_on_simplex(new_p))
        return
    
    def check_stop(self, s):
        if s%10==0: print("Iteration " + str(s))
        if s < self.max_iter:
            if s > 0 and not self.check_epsilon():
                self.stop_message = "In " + str(s) + " iterations, the minimum displacement has been reached."
                return False
            return True
        self.stop_message = "The maximum number of iterations has been reached."
        return False
    
    def check_epsilon(self):
        if  np.linalg.norm(self.p_list[-1] - self.p_list[-2]) < self.eps: return False
        return True

    def gradient_p(self, p):

        evaluations_on_training = self.evaluate_on_training(p)
        gradient = np.zeros((self.K))
        if self.KL_type == 0:
            for k in range(self.K):
                for i in range(self.K):
                    gradient[k] += np.sum(self.constant_part[i] * self.get_KDE_evaluations_on_training(k,i) / evaluations_on_training[i])/self.n[i]
        else:
            for k in range(self.K):
                for i in range(self.K):
                    gradient[k] += np.sum( self.constant_part[i] * self.get_KDE_evaluations_on_training(k,i) * (self.U[i] + 1 + np.log(evaluations_on_training[i])) )/self.n[i]

        return gradient/self.K
    
    ####====---- Estimator point evaluation fucntions ----====####

    def evaluate_on_training(self, p): # Evaluates the $p$-weighted modes KDE in the training points; Returns a list of size K whose $i$-th entry is $(\langle p, \hat{\nu} \rangle (x_{.,i}))$. This output helps computing the surrogate function and its derivative
        outputs = []
        for i in range(self.K):
            output = np.zeros((self.n[i]))
            for j in range(self.K):
                output += p[j] * self.get_KDE_evaluations_on_training(j, i)
            outputs.append(output)
        return outputs
    
    def evaluate(self, X_eval, p=None):
        """
        Evaluates the density made of the KDE of the modes linearly combined by the weight p.

        Parameters
        ----------
        X_eval: np.ndarray of shape (n_eval,d)
        The datapoints in which the density estimator must be evaluated

        p: np.ndarray of shape (K)
        The parameter $p$ used to define the estimator. Default is the last parameter in the attribute p_list.

        Returns
        -------
        np.ndarray of shape (n_eval)
        The corresponding density values $\\langle p, \\hat{\\nu}(X_eval) \\rangle$.
        """
        if p is None: p = self.p_list[-1]
        n = X_eval.shape[0]
        output = np.zeros((n))
        for i in range(self.K):
            output += p[i] * KDE_evaluation(self.X[i], X_eval, self.alpha[i], self.n_periods)
        return output
    
    ####====---- From surrogate to objective functions ----====####

    def surrogate_function(self, p=None):
        if p is None: p = self.p_list[-1]
        evaluations_on_training = self.evaluate_on_training(p)
        val = 0
        if self.KL_type==0:
            for i in range(self.K):
                val += np.sum(np.log(evaluations_on_training[i]) * self.constant_part[i])/self.n[i]
        else:
            for i in range(self.K):
                val += np.sum(evaluations_on_training[i] * ( self.U[i] + np.log(evaluations_on_training[i]) ) * self.constant_part[i])/self.n[i]
        return val/self.K
    
    def surrogate_to_objective_1(self, nu, p=None):
        if p is None: p = self.p_list[-1]
        evaluations_on_training = self.evaluate_on_training(p)
        val = 0
        if self.KL_type==0:
            for i in range(self.K):
                val += np.sum( - np.log(evaluations_on_training[i]) * np.exp(-self.U[i]) / nu[i](self.X[i]))/self.n[i]
        else:
            for i in range(self.K):
                val += np.sum(evaluations_on_training[i] * ( self.U[i] + np.log(evaluations_on_training[i]) ) / nu[i](self.X[i]))/self.n[i]
        return val/self.K
    
    def surrogate_to_objective_2(self, nu, p=None):
        if p is None: p = self.p_list[-1]
        ideal_p_evaluations_on_training = []
        for i in range(self.K):
            output = np.zeros((self.n[i]))
            for r in range(self.K):
                output += p[r] * nu[r](self.X[i])
            ideal_p_evaluations_on_training.append(output)

        val = 0
        if self.KL_type==0:
            for i in range(self.K):
                val += np.sum( - np.log(ideal_p_evaluations_on_training[i]) * np.exp(-self.U[i]) / nu[i](self.X[i]))/self.n[i]
        else:
            for i in range(self.K):
                val += np.sum(ideal_p_evaluations_on_training[i] * ( self.U[i] + np.log(ideal_p_evaluations_on_training[i]) ) / nu[i](self.X[i]))/self.n[i]
        return val/self.K
    
    def objective_function(self, U, nu, X_quad, p=None): # This is the ideal objective function that we would like to minimise: J_\infty(p). Evaluates the quality of p
        if p is None: p = self.p_list[-1]
        n_quad = X_quad.shape[0]
        estimate_values = np.zeros((n_quad))
        for r in range(self.K):
            estimate_values += p[r] * nu[r](X_quad)
        if self.KL_type==0:
            return - np.sum(np.exp(-U(X_quad)) * np.log(estimate_values))/n_quad
        else:
            return np.sum(estimate_values * (np.log(estimate_values) + U(X_quad)))/n_quad
        
    def objective_function_measure(self, U, X_quad, p=None):# This is the objective function proportional to the KL, which takes as an input the estimated density
        if p is None: p = self.p_list[-1]
        n_quad = X_quad.shape[0]
        estimate_values = self.evaluate(X_quad, p)
        if self.KL_type==0:
            return - np.sum(np.exp(-U(X_quad)) * np.log(estimate_values))/n_quad
        else:
            return np.sum(estimate_values * (np.log(estimate_values) + U(X_quad)))/n_quad

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

########========-------- TESTS --------========########


"""
# Target preparation:

n_center = 2
dim = 1
sigma_min, sigma_max = 0.08, 0.08
centers = np.array([np.array([0.25]), np.array([0.75])]) #np.random.uniform(-5, 5, (n_center, dim))
sigma_star = np.random.uniform(sigma_min, sigma_max , (n_center))
c_star = np.array([0.75, 0.25])#np.random.uniform(0,1, (n_center))
p_star = c_star/np.sum(c_star)

def U(X):# X: (n,d)
    density_value = np.zeros((len(X)))
    for i in range(n_center):
        density_value += p_star[i] * compute_gaussian_density(X - centers[i], sigma_star[i])
    return -np.log(density_value)

def nu_1(X):
    return compute_gaussian_density(X - centers[0], sigma_star[0])

def nu_2(X):
    return compute_gaussian_density(X - centers[1], sigma_star[1])

nu = [nu_1, nu_2]

sample_repartition = np.ones((n_center))/n_center # equal
n_tot = 1600
arr = sample_repartition * n_tot
out = np.empty_like(arr, dtype=np.int64)
n_samples = np.ceil(arr, out, casting='unsafe')
n_tot = np.sum(n_samples) # corrected due to thresholding
X_samples, U_samples = [], []
for i in range(n_center):
    raw_samples = np.random.normal(0, sigma_star[i], (n_samples[i],dim)) + centers[i]
    X_samples.append(np.mod(raw_samples, np.ones((n_samples[i],dim)) ))
    U_samples.append(U(X_samples[-1]))

# Algorithm parametrisation:

p0 = np.ones((n_center))/n_center
h = 0.1
max_iter = 400
eps = 0.0001
KL_type = 1
bw = ['silverman']*n_center

solver = GD_estimator(KL_type, p0, U_samples, X_samples, h, eps, max_iter, bw)

solver.optimise()

#print('Bandwidth selection: ' + str(solver.alpha))
print('Ideal weights: ' + str(p_star))
print('Estimated weights: ' + str(solver.p_list[-1]))
#print('Modes standard deviation: ' + str(sigma_star))
#print('Surrogate evaluation at the beginning of the GD: ' + str(solver.surrogate_function(solver.p_list[0])))
#print('Surrogate evaluation at the end of the GD: ' + str(solver.surrogate_function()))
#X_quad = 6 * ((np.arange(400)/400).reshape(-1, 1)*2-1)
#print('Real target evaluation at the end of the GD: ' + str(solver.objective_function(U, nu, X_quad)))

#########################################################################################################################

# Graph display

alpha = solver.alpha

fig,ax = plt.subplots(2,2)

X_eval = (np.arange(400)/400).reshape(-1, 1)
estimated_modes = []
real_modes = []
for i in range(n_center):
    real_modes.append(nu[i](X_eval))
    estimated_modes.append(KDE_evaluation(X_samples[i], X_eval, alpha[i]))
    ax[0][0].plot(X_eval, real_modes[-1], label='nu_'+str(i))
    ax[0][0].plot(X_eval, estimated_modes[-1], label='estimated nu_'+str(i))

ax[0][0].set_title('Mode densities estimated')
ax[0][0].legend()

y_1 = []
for i in range(len(solver.p_list)):
    val = solver.surrogate_function(solver.p_list[i])
    y_1.append(val)
ax[0][1].plot(np.arange(len(solver.p_list)), y_1, label='Surrogate Objective function')
ax[0][1].set_title('Objective function values along the descent')
ax[0][1].legend()

#y_2=[]
#for i in range( M ):
#    y_2.append(KL_wrt_mu(solver, U, solver.p_list[i%len(solver.p_list)], solver.sigma_list[i%len(solver.sigma_list)]))
#ax[2][1].plot(np.arange(M), y_2, label='Ideal objective function')
#ax[2][1].set_title('Objective function values along the descent')
#ax[2][1].legend()

for i in range(n_center):
    ax[1][0].plot(np.arange(len(solver.p_list)), np.array(solver.p_list)[:,i], label='p_'+str(i))
ax[1][0].set_title('Weight values along the descent')
ax[1][0].legend()

y = solver.evaluate(X_eval, solver.p_list[-1])
z = solver.evaluate(X_eval, solver.p_list[0])
ax[1][1].plot(X_eval, y, label='final estimation')
ax[1][1].plot(X_eval, z, label='initial estimation')
ax[1][1].plot(X_eval, np.exp(-U(X_eval)), label='True distribution')
ax[1][1].set_title('Estimation vs reality')
ax[1][1].legend()

plt.tight_layout()
plt.show()
"""

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################