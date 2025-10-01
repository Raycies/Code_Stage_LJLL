import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
# KDE packages (used only for their bandwidth selectors):
from statsmodels.nonparametric import kernel_density
from KDEpy import NaiveKDE

########========--------- Bandwidth selection methods --------========########

def get_bandwidth(X, bw):
    # Wrapper for the different bandwidth selectors
    type_bw = type(bw)
    bw_dictionary = {'silverman':compute_bandwidth_silverman, 'cv_ml':compute_bandwidth_cv_ml, 'cv_ls':compute_bandwidth_cv_ls, 'isj':compute_bandwidth_isj}
    if type_bw == float:
        return bw
    else:
        if bw not in bw_dictionary.keys():
            raise ValueError('The bandwidth bw given is not among the accepted values')
        return bw_dictionary[bw](X)
    
def compute_bandwidth_silverman(X):
    n, d = X.shape
    if d==1:
        XX = X[:,0]
        hat_sigma = np.std(XX)
        multiplicative_factor = 1.06 # 1.06, 0.9 can also be chosen
        return multiplicative_factor * min(hat_sigma, (np.quantile(XX, 0.75) - np.quantile(XX,0.25))/1.34)*n**(-1/5)
    else:
        mean = np.mean(X, axis=0)
        sq_std = np.sum(np.linalg.norm(X - mean, axis=1)**2)/(n*d)
        return (4/(d+2))**(1/(d+4)) * np.sqrt(sq_std) * n**(-1/(d+4))
    
def compute_bandwidth_cv_ml(X):
    n, d = X.shape
    estimator = kernel_density.KDEMultivariate(X, 'c'*d, bw='cv_ml')
    bw = estimator.bw # Provides a bandwidth for each coordinate (bandwidth matrix of diagonal form)
    return np.mean(bw)

def compute_bandwidth_cv_ls(X):
    n, d = X.shape
    estimator = kernel_density.KDEMultivariate(X, 'c'*d, bw='cv_ls')
    bw = estimator.bw # Provides a bandwidth for each coordinate (bandwidth matrix of diagonal form)
    return np.mean(bw)

def compute_bandwidth_isj(X):
    n, d = X.shape
    if d!=1:
        raise Exception('The improved Sheather-Jones selector is only available for dimension 1')
    else:
        estimator = NaiveKDE(kernel='gaussian', bw='ISJ').fit(X)
        return estimator.bw

########========--------- Regularisers --------========########

def unit_ball_volume(d):
    if d > 3: return math.pi**(d/2)/math.gamma(d/2 + 1)
    if d==1: return 2
    if d==2: return 4*np.pi
    if d==3: return np.pi*4/3

def compute_reg_value(GD_object, reg_type, p): # Wrapper to guide to the adequate regulariser for reg_type.
    reg_type_dictionary = {'l_2':value_l_2, 'h_1':value_h_1}
    if reg_type not in reg_type_dictionary.keys():
        raise ValueError('The bandwidth bw given is not among the accepted values')
    return reg_type_dictionary[reg_type](GD_object, p)

def value_l_2(GD_object, p):
    return np.sum(p**2)

def value_h_1(GD_object, p):
    P = np.tile(p, (GD_object.n,1))
    return np.sum((P - P.T)**2 * GD_object.penalty_constant_matrix)

def compute_reg_gradient(GD_object, reg_type, p):# Wrapper to guide to the adequate gradient for reg_type.
    reg_type_dictionary = {'l_2':gradient_l_2, 'h_1':gradient_h_1}
    if reg_type not in reg_type_dictionary.keys():
        raise ValueError('The bandwidth bw given is not among the accepted values')
    return reg_type_dictionary[reg_type](GD_object, p)

def gradient_l_2(GD_object, p): # Gradient associated to the regulariser $n\sum_{i=1}^n p_i^2
    return len(p)*2*p

def gradient_h_1(GD_object, p): # Gradient associated to the regulariser $\sum_{i,j=1}^n (p_i - p_j)^2 \frac{ \un_{\|x_i-x_j\| \leq b_n} }{\max(a_n^2, \|x_i-x_j\|^2) |B(0,b_n)|}
    P = np.tile(p, (GD_object.n,1))
    return 4 * np.sum( (P - P.T) * GD_object.penalty_constant_matrix, axis=0)


########========--------- Gaussian computing methods --------========########

def rho(sq_distances, sigma, d): # computes the gaussian density
    return np.exp(-sq_distances/(2*sigma**2))/(sigma**d * (2*np.pi)**(d/2))

def KDE_evaluation(X_train, X_eval, sigma): # Computes \hat{\nu}(x_{.,eval}), ie the KDE based on X_train, evaluated at points X_eval
    n, d = X_train.shape
    sq_D = euclidean_distances(X_eval, X_train, squared=True)
    return np.sum(rho(sq_D, sigma, d), axis=1) / n

########========--------- Projection on simplex --------========########

def projection_on_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

########========--------- Gradient descent and estimator class --------========########

class GD_estimator:

    def __init__(self, KL_type, p0, U, X, h, eps, max_iter, bw, reg_types, reg_constants):
        """
        Computes the KDE bandwidth and then the values of the KDE in the datapoints, which will be used at each step of the gradient descent. There is no output at this step. The bandwidth is kept in the attribute 'alpha'.

        Parameters
        ----------
        KL_type: integer, either 0 or 1
        If 0, the algorithm based on $\pi \mapsto KL(\mu, \pi)$ is used. If 1 it is $\pi \mapsto KL(\pi|\mu)$, \mu being the target and \pi the estimate.

        p0: np.ndarray of shape (n)
        Initial value of the weight vector for the gradient descent, must belong to the simplex $\Delta_{n-1}$.

        U: np.ndarray of shape (n)
        Gibbs potential values of each datapoint.

        X: np.ndarray of size (n,d)
        Datapoints of dimension d corresponding to U.

        h: float
        Gradient descent time step.

        eps: float
        A stopping criteria for the gradient descent. The descent ends when the euclidean distance between two consecutive weights is lower than 'eps' or if another stopping criteria got fulfilled earlier.

        max_iter: int
        A stopping criteria for the gradient descent. The descent ends when the number of time steps reaches max_iter or if another stopping criteria got fulfilled earlier.

        bw: string, float
        Bandwidth selection rule for the sampled density KDE. The valid values are: a positive float (the bandwidth is fixed at this value); 'silverman' for the Silverman's rule of thumb which is appropriate for unimodal data and optimal in the AMISE sense for Gaussian data; 'cv_ml' for the cross validation maximum likelihood selection rule, which works for multimodal data; 'cv_ls' for the cross validation maximum least squares selection rule, which works for multimodal data; 'isj' for the improved Sheather Jones plug-in selection rule introduced by Botev et al. which only works for one dimensional data.

        reg_types: Sequence[string]
        List of regularisation terms to add to the surrogate function. The valid values are: 'l_2' for the regularisation term $n\sum_{i=1}^n p_i^2$; 'h_1' for the regularisation term of the form $\sum_{i,j=1}^n (p_i-p_j)^2 c(x_i, x_j, a_n, b_n)$.

        reg_constants: Sequence[float]
        List of positive multiplicative constants corresponding to the regularisation terms. Must be of the same length as reg_types.

        Returns
        -------
        None (the computed bandwidth parameters are stored in the attribute alpha)
        """
        self.KL_type = KL_type
        self.d = len(X[0]) # Space dimension
        self.n = len(X) # number of datapoints
        self.bw = bw
        self.alpha = get_bandwidth(X, bw) # Bandwidth storage
        self.h = h
        self.U = U
        self.X = X
        self.reg_types = reg_types
        self.reg_constants = reg_constants
        self.p_list = [p0]
        self.max_iter = max_iter
        self.eps = eps
        self.stop_message = ""

        self.sq_distances = euclidean_distances(self.X, self.X, squared=True)
        self.kernel_distances = rho(self.sq_distances, self.alpha, self.d)

        if self.KL_type==0:
            self.constant_part = - np.exp(-U)/ ( np.sum(self.kernel_distances, axis=0) / self.n ) # list of $- e^{-U(x_i)}/\hat{\nu}_{\alpha}(x_i)$, this is a constant coefficient along the descent
        else:
            self.constant_part = 1 / ( np.sum(self.kernel_distances, axis=0) / self.n ) # list of $1/\hat{\nu}_{\alpha}(x_i)$, this is a constant coefficient along the descent

        self.penalty_constant_matrix = self.compute_constant_penalty_matrix()
              
        return

    def get_sq_distances(self, i, j): # returns pair-wise data squared distances of $X_i, X_j$, a 2D array of shape (n_i, n_j)
        if i <= j: return self.sq_distances[j + i*self.K]
        else: return self.sq_distances[i + j*self.K].T

        ####====---- Utilitary ----====####


    def compute_constant_penalty_matrix(self):
        low_threshold = 1/np.sqrt(self.n)
        high_threshold = 1/(self.n**(1/4))
        sq_distances = self.sq_distances
        clipped_sq_distances = np.clip(sq_distances, low_threshold**2, None)
        elements_taken = (sq_distances <= high_threshold**2).astype(int)
        print(elements_taken.shape)
        print(clipped_sq_distances.shape)
        return elements_taken/(clipped_sq_distances * unit_ball_volume(self.d))

        #low_threshold = 1/self.n
        #high_threshold = 1/(self.n**(1/2))
        #return 1/(np.clip(self.sq_distances, low_threshold, high_threshold) * (high_threshold**self.d - low_threshold**self.d) * unit_ball_volume(self.d) )
    
        #Last threshold matrix (worked rather well):
        # Other threshold matrix
        #threshold = 1/self.n
        #return 1/np.clip(self.sq_distances, threshold, None)


    ####====---- Gradient descent functions ----====####

    def descent_step(self):
        old_p = self.p_list[-1]
        new_p_before_projection = old_p - self.h * self.gradient_p(old_p)
        self.high_values_test(new_p_before_projection)
        self.p_list.append( projection_on_simplex( new_p_before_projection ) )
    
        return
    
    def optimise(self):
        """
        Runs the gradient descent on the surrogate function of $\Delta_{n-1}$

        Parameters
        ----------
        None

        Returns
        -------
        None (the successive weights computed are available in the attribute p_list which is a list of np.ndarray of shape (n))
        """
        s=0
        self.keep_going = True
        while self.check_stop(s):
            self.descent_step()
            s+=1
        print("Gradient descent terminated ; " + self.stop_message)
        return
    
    def check_stop(self, s):
        if s%5==0: print("Iteration " + str(s)) # Display
        if self.keep_going == False: return False # If something went wrong during the descent, it stops there
        if s < self.max_iter:
            if s > 0 and not self.check_epsilon():
                self.stop_message = "In " + str(s) + " iterations, the minimum displacement has been reached."
                self.keep_going = False
        else:
            self.stop_message = "The maximum number of iterations has been reached."
            self.keep_going = False
        
        return self.keep_going
    
    def high_values_test(self, p):
        if np.sum(p) > 1e10: # If this happens, points risk to jump from one end of the simplex to another, this is not the smooth behaviour expected from a gradient descent.
            self.stop_message = 'The values taken by the weight before the projection onto the simplex are too high. Consider lowering the gradient descent time steps.'
            self.keep_going = False
        return False
    
    def check_epsilon(self):
        if np.linalg.norm(self.p_list[-1] - self.p_list[-2]) < self.eps: return False
        return True

    def gradient_p(self, p):
        gradient = np.zeros((self.n))
        pi_values = np.sum(self.kernel_distances * np.tile(p, (self.n,1)), axis=1 ) # (\pi(x_i))_{i \in \{1,...,n\}}
        if self.KL_type==0:
            new_constant = self.constant_part / pi_values
            gradient += np.sum(self.kernel_distances * np.tile(new_constant, (self.n,1)), axis=1)/self.n
            
        else:
            new_constant = self.constant_part * (self.U + 1 + np.log(pi_values))
            gradient += np.sum(self.kernel_distances * np.tile(new_constant, (self.n,1)), axis=1)/self.n

        for i in range(len(self.reg_types)):
            reg_type, reg_cst = self.reg_types[i], self.reg_constants[i]
            gradient += reg_cst * compute_reg_gradient(self, reg_type, p)
        return gradient
    
    ####====---- Estimator point evaluation functions ----====####

    def evaluate(self, X_new, p):
        """
        Evaluates the estimated density with weight parameter p.

        Parameters
        ----------
        X_new: np.ndarray of shape (n_eval,d)
        The datapoints in which the density estimator must be evaluated

        p: np.ndarray of shape (K)
        The parameter $p$ used to define the estimator. Default is the last parameter in the attribute p_list.

        Returns
        -------
        np.ndarray of shape (n_eval)
        The corresponding estimated density values.
        """
        return np.sum( rho(euclidean_distances(X_new, self.X, squared=True), self.alpha, self.d) * np.tile(p, (len(X_new),1)), axis=1 )   
    
    ####====---- From surrogate to objective functions ----====####

    def surrogate_function(self, p):
        P = np.tile(p, (self.n,1))
        objective_without_reg = np.sum(self.constant_part * np.log(np.sum(self.kernel_distances * np.tile(p, (self.n,1)), axis=1 )))/self.n
        reg_values = []
        for i in range(len(self.reg_types)):
            reg_type, reg_cst = self.reg_types[i], self.reg_constants[i]
            reg_values.append(reg_cst * compute_reg_value(self, reg_type, p))
        return objective_without_reg, reg_values


    
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

########========-------- TESTS --------========########


"""# Target preparation:

def g(X, sigma): # X:(n,d) -> we want as an output smth of size (n) with ith component g(x[i]) 
    d = len(X[0])
    return np.exp(-np.linalg.norm(X, axis=1)**2/(2*sigma**2))/(sigma**d * (2*np.pi)**(d/2))

def sample_from_nu(n_samples, p_sampling, sampling_centers, sigma_samples, d):
    
    arr = p_sampling * n_samples
    out = np.empty_like(arr, dtype=np.int64)
    n_samples_repartition = np.ceil(arr, out, casting='unsafe')
    n_tot = np.sum(n_samples_repartition)
    X = []
    for i in range(len(n_samples_repartition)):
        X.append(np.random.normal(0, sigma_samples[i], (n_samples_repartition[i], d)) + sampling_centers[i])
    return np.concatenate(X, axis=0)


dim = 1

sampling_centers = np.array([np.array([-0.8]), np.array([1.1])])
n_sampling_centers = len(sampling_centers)
sigma_samples = [0.8, 0.8]
c_sampling = np.array([1, 1])
p_sampling = c_sampling/np.sum(c_sampling)

true_centers = np.array([np.array([-1]), np.array([1])])
n_true_centers = len(true_centers)
sigma_star = [0.6, 0.6]
c_star = np.array([2,1])
p_star = c_star/np.sum(c_star)

def U(X):# X: (n,d)
    density_value = np.zeros((len(X)))
    for i in range(n_true_centers):
        density_value += p_star[i] * g(X - true_centers[i], sigma_star[i])
    return -np.log(density_value)

def nu(X):
    density_value = np.zeros((len(X)))
    for i in range(n_sampling_centers):
        density_value += p_sampling[i] * g(X - sampling_centers[i], sigma_samples[i])
    return density_value

n_samples = 800

X_samples = sample_from_nu(n_samples, p_sampling, sampling_centers, sigma_samples, dim)
U_samples = U(X_samples)

# Algorithm parametrisation:

p0 = np.ones((n_samples))/n_samples
h = 0.0001
max_iter = 100
eps = 0.0001
KL_type = 0
bw = 'cv_ml'
reg_types = ['h_1']
reg_constants = [0.0001]


solver = GD_estimator(KL_type, p0, U_samples, X_samples, h, eps, max_iter, bw, reg_types, reg_constants)

solver.optimise()

# Display

alpha = solver.alpha

fig,ax = plt.subplots(2,2)

X_eval = 6 * ((np.arange(400)/400).reshape(-1, 1)*2-1)
estimated_modes = KDE_evaluation(X_samples, X_eval, alpha)
real_modes = nu(X_eval)
ax[0][0].plot(X_eval, real_modes, label='True nu')
ax[0][0].plot(X_eval, estimated_modes, label='Estimated nu')
ax[0][0].set_title('Samples density estimated')
ax[0][0].legend()

y_1 = []
for i in range(len(solver.p_list)):
    val = solver.surrogate_function(solver.p_list[i])[0]
    y_1.append(val)
ax[0][1].plot(np.arange(len(solver.p_list)), y_1, label='Surrogate Objective function')
ax[0][1].set_title('Objective function values along the descent')
ax[0][1].legend()

for i in range(n_samples):
    if i%5==0:# One every 5 is displayed
        ax[1][0].plot(np.arange(len(solver.p_list)), np.array(solver.p_list)[:,i], label='p_'+str(i))
ax[1][0].set_title('Weight values along the descent')

y = solver.evaluate(X_eval, solver.p_list[-1])
z = np.exp(-U(X_eval))
ax[1][1].plot(X_eval, y, label='final estimation')
ax[1][1].plot(X_eval, z, label='True distribution')
ax[1][1].scatter(X_samples, solver.p_list[-1] * max(z) / max(solver.p_list[-1]))
ax[1][1].set_title('Estimation vs reality')
ax[1][1].legend()

plt.tight_layout()
plt.show()"""

