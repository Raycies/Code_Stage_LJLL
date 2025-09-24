import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from timeit import default_timer as timer
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


########========--------- Gaussian computing methods --------========########

def rho(sq_distances, sigma, d): # computes the gaussian density (N(0, sigma^2 I_d))
    return np.exp(-sq_distances/(2*sigma**2))/(sigma**d * (2*np.pi)**(d/2))

def KDE_evaluation(X_train, X_eval, sigma): # Computes \hat{\nu}(x_{.,eval}), ie the KDE based on X_train, evaluated at points X_eval
    n, d = X_train.shape
    sq_D = euclidean_distances(X_eval, X_train, squared=True)
    return np.sum(rho(sq_D, sigma, d), axis=1) / n

########========--------- Projection on simplex --------========########

def projection_on_simplex(x): # Michelot (1986)
    # Input: $x$ a vector
    # Output: $P(x)$ the projection of $x$ on the set of admissible directions for the simplex $\Delta_{K-1}$
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(x)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0.0)


########========--------- Objective function, surrogate functions --------========########

# DEPRECATED, but I don't erase it bc I don't remember whats up with the masks

def find_mask(X,Y): # at the end: Xlog(Y). Mask when log(Y) is inaccurate, ie when Y is too low
    return Y >= 1e-200

def KL_right_quadrature(U, solver, p, sigma, X_quad): # Computes the objective function infinite data
    # Numerical issue: The exponential may be very low (e-50), due to numerical error the log term will be worth -inf. To avoid the result being -inf, we select only the indexes such that the exponential term is higher than a threshold.
    exp_part = np.exp(-U(X_quad))
    estimator_values = solver.estimated_density(X_quad, p, sigma)
    mask = find_mask(exp_part, estimator_values)
    return - np.sum(exp_part[mask] * np.log(estimator_values[mask])) / np.sum(mask)

def J_infty_right_quadrature(U, nu, p, X_quad): #Computes J_\infty(p)
    exp_part = np.exp(-U(X_quad))
    estimator_values = np.zeros((len(X_quad)))
    for i in range(len(nu)):
        estimator_values += p[i] * nu[i](X_quad)
    mask = find_mask(exp_part, estimator_values)
    return - np.sum(exp_part[mask] * np.log(estimator_values[mask]/exp_part[mask])) / np.sum(mask)

def true_KL_for_estimator(U, solver, p, sigma, X_quad): # Computes the True KL (works only when e^-U sums to 1)
    # Numerical issue: The exponential may be very low (e-50), due to numerical error the log term will be worth -inf. To avoid the result being -inf, we select only the indexes such that the exponential term is higher than a threshold.
    exp_part = np.exp(-U(X_quad))
    estimator_values = solver.estimated_density(X_quad, p, sigma)
    mask = find_mask(exp_part, estimator_values)
    return - np.sum(exp_part[mask] * np.log(estimator_values[mask]/exp_part[mask])) / np.sum(mask)

########========--------- Gradient descent and estimator class --------========########

class GD_estimator:

    def __init__(self, KL_type, p0, U, X, h, eps, max_iter, bw):
        # Input: - KL_type, an integer. If 0, it is KL(\mu, \pi) if 1 it is KL(\pi|\mu) (\mu being the target and \pi the estimate)
        #        - $p_0$ the initial values of the weight vector $p$
        #        - U, X tuples of $K$ numpy arrays, containing the initial data
        #        - $h$ is the time discretisation step
        #        - eps is a stopping criteria, if the distance between two time steps is lower than eps, we stop the descent (eps[0]: for p, eps[1]: for sigma)
        #        - max_iter is a stopping criteria, if this number of descent iterations is reached, we stop
        #        - bw is the selector used to compute once and for all the bandwidth used for KDE. valid values: float (fixed bandwidth), 'silverman', 'cv_ml', 'cv_ls', 'isj' ('silverman' is the optimal AMISE bandwidth for Gaussian density, 'cv_ml' optimiser of cross validation maximum likelihod, 'cv_ls' optimiser of cross_validation integrated square error, 'isj' improved sheather Jones, plug-in method introduced by Botev et al. in KDE via diffusion paper (only in 1D)). 'silverman' is a good option when the data is unimodal, otherwise 'cv_ml' and 'cv_ls' work rather well, but they take much more bandwidth computation time. Must be a list of what is used for each mode, e.g. with d=3: ['cv_ml', 'silverman', 'cv_ml'].

        self.d = len(X[0][0])
        self.K = len(X) # number of clusters
        self.n = [0]*self.K # number of data in each cluster, filled later in a for loop
        self.alpha = [0]*self.K # bandwidth storage for each mode, filled later in a for loop
        self.h = h
        self.U = U
        self.X = X
        self.KL_type = KL_type
        self.max_iter = max_iter
        self.eps = eps
        self.stop_message = ""
        self.bw = bw

        self.KDE_evaluations_on_training = {} # The fact that sigma does not vary allows us to keep only the sum values. self.get_gaussian_sums(i,j) is a numpy array of size n_j, whose q-th component is: \frac{1}{n_i} \sum_{k=1}^{n_i} g_{\sigma_i}(x_{k,i} - x_{q,j}). Ie self.get_gaussian_sums(i,j) = (\hat{\nu}_i(x_{q,j}))_{q \in \{1,...,n_j\}}
        print('Starting bandwidth computations and KDE evaluations in the training points...')
        self.start_timer()
        for k in range(self.K):
            self.n[k] = len(X[k])
            self.alpha[k] = get_bandwidth(X[k], bw[k]) # KDE in the torus
            for q in range(self.K):
                self.KDE_evaluations_on_training[q + k*self.K] = KDE_evaluation(self.X[k], self.X[q], self.alpha[k]) # Computes \hat{\nu}_k(x_{.,q})
        self.stop_timer()
        print("Done. Took "+str(round(self.stop_time - self.start_time,2))+"s.")
        self.constant_part = [] # list of $-e^{-U(x_{j,i})}/\hat{\nu}_{\alpha_i}(x_{j,i})$ (when KL_type=0), this is a constant coefficient along the descent
        for k in range(self.K):
            hat_nu_k = self.get_KDE_evaluations_on_training(k,k)

            if self.KL_type==0: self.constant_part.append( - np.exp(-U[k]) / hat_nu_k )
            else: self.constant_part.append( 1 / hat_nu_k )

        self.p_list = [p0]
        return
    
    def get_KDE_evaluations_on_training(self, i, j): # returns \hat{\nu}_i(x_{.,j})
        return self.KDE_evaluations_on_training[j + i*self.K]
    
    ####====---- Gradient descent functions ----====####

    def descent_step(self):
        old_p = self.p_list[-1]
        self.p_list.append( projection_on_simplex( old_p - self.h * self.gradient_p(old_p) ) )
        return
    
    def start_timer(self):
        self.start_time = timer()
        return
    
    def stop_timer(self):
        self.stop_time = timer()
    
    def optimise(self):
        s=0
        self.start_timer()
        while self.check_stop(s):
            self.descent_step()
            s+=1
        self.stop_timer()
        print("Gradient descent done ; " + self.stop_message)
        print("Took "+str(round(self.stop_time - self.start_time,2))+"s.")
        return
    
    def check_stop(self, s, one_every=20): # One every one_every iterations, a message is displayed.
        if s%one_every==0: print("Iteration " + str(s))
        if s < self.max_iter:
            if s > 0 and not self.check_epsilon():
                self.stop_message = "In " + str(s) + " iterations, the minimum displacement has been reached."
                return False
            return True
        self.stop_message = "The maximum number of iterations has been reached."
        return False
    
    def check_epsilon(self):
        if np.linalg.norm(self.p_list[-1] - self.p_list[-2]) < self.eps: return False
        else: return True

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
    
    ####====---- Estimator point evaluation functions ----====####
    
    def evaluate_on_training(self, p): # Evaluates the $p$-weighted modes KDE in the training points; Returns a list of size K whose $i$-th entry is $(\langle p, \hat{\nu} \rangle (x_{.,i}))$. This output helps computing the surrogate function and its derivative
        outputs = []
        for i in range(self.K):
            output = np.zeros((self.n[i]))
            for j in range(self.K):
                output += p[j] * self.get_KDE_evaluations_on_training(j, i)
            outputs.append(output)
        return outputs
    
    def evaluate(self, X_eval, p=None): # Evaluates the $p$-weighted modes KDE in X_eval. Default is the last value of $p$ in $p_list$
        if p is None: p = self.p_list[-1]
        n = X_eval.shape[0]
        output = np.zeros((n))
        for i in range(self.K):
            output += p[i] * KDE_evaluation(self.X[i], X_eval, self.alpha[i])
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
    
    def objective_function(self, U, nu, X_quad, p=None): # This is the ideal objective function that we would like to minimise (J_\infty(p))
        if p is None: p = self.p_list[-1]
        n_quad = X_quad.shape[0]
        estimate_values = np.zeros((n_quad))
        for r in range(self.K):
            estimate_values += p[r] * nu[r](X_quad)
        if self.KL_type==0:
            return - np.sum(np.exp(-U(X_quad)) * np.log(estimate_values))/n_quad
        else:
            return np.sum(estimate_values * (np.log(estimate_values) + U(X_quad)))/n_quad
        
    def KL(self, U, X_quad, p=None):
        if p is None: p = self.p_list[-1]
        n_quad = X_quad.shape[0]
        estimate_values = np.zeros((n_quad))
        for r in range(self.K):
            estimate_values += p[r] * KDE_evaluation(self.X[r], X_quad, self.alpha[r])
        if self.KL_type==0:
            return - np.sum(np.exp(-U(X_quad)) * np.log(estimate_values))/n_quad
        else:
            return np.sum(estimate_values * (np.log(estimate_values) + U(X_quad)))/n_quad

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

########========-------- TESTS --------========########


# Target preparation:

"""n_center = 2
dim = 1
sigma_min, sigma_max = 0.5, 0.50001
centers = np.array([np.array([-1]), np.array([1])]) #np.random.uniform(-5, 5, (n_center, dim))
sigma_star = 0.6, 0.6 #np.random.uniform(sigma_min, sigma_max , (n_center))
c_star = np.array([0.75, 0.25])#np.random.uniform(0,1, (n_center))
p_star = c_star/np.sum(c_star)

def g(X, sigma): # X: (n,d) -> we want as an output smth of size (n) with ith component g(x[i]) 
    d = len(X[0])
    return np.exp(-np.linalg.norm(X, axis=1)**2/(2*sigma**2))/(sigma**d * (2*np.pi)**(d/2))



def U(X):# X: (n,d)
    density_value = np.zeros((len(X)))
    for i in range(n_center):
        density_value += p_star[i] * g(X - centers[i], sigma_star[i])
    return -np.log(density_value)

def nu_1(X):
    return g(X - centers[0], sigma_star[0])

def nu_2(X):
    return g(X - centers[1], sigma_star[1])

nu = [nu_1, nu_2]

sample_repartition = np.ones((n_center))/n_center # equal
n_tot = 8
arr = sample_repartition * n_tot
out = np.empty_like(arr, dtype=np.int64)
n_samples = np.ceil(arr, out, casting='unsafe')
n_tot = np.sum(n_samples) # corrected due to thresholding
X_samples, U_samples = [], []
for i in range(n_center):
    X_samples.append( np.random.normal(0, sigma_star[i], (n_samples[i],dim)) + centers[i] )
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

X_eval = 6 * ((np.arange(400)/400).reshape(-1, 1)*2-1)
estimated_modes = []
real_modes = []
for i in range(n_center):
    real_modes.append(g(X_eval - centers[i], sigma_star[i]))
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
plt.show()"""


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################