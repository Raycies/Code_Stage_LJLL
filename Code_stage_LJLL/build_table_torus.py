import matplotlib.pyplot as plt
import numpy as np
import mode_weights_torus
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
storage_folder = os.path.join(script_dir, "table_storage")

file_name_output = 'data_collection_torus'

#### Target Gibbs measure definition ####

n_center = 2
dim = 1
centers = np.array([np.array([0.25]), np.array([0.75])])
sigma_star = 0.15 * np.ones((n_center))
p_star = np.array([0.75, 0.25])

def U(X):# X: (n,d)
    density_value = np.zeros((len(X)))
    for i in range(n_center):
        density_value += p_star[i] * mode_weights_torus.compute_gaussian_density(X - centers[i], sigma_star[i])
    return -np.log(density_value)

def nu_1(X):
    return mode_weights_torus.compute_gaussian_density(X - centers[0], sigma_star[0])

def nu_2(X):
    return mode_weights_torus.compute_gaussian_density(X - centers[1], sigma_star[1])

nu = [nu_1, nu_2]

#### Sampling ####

n_dataset = 10
n_tot = 2400
sample_repartition = np.ones((n_center))/n_center # equal
arr = sample_repartition * n_tot
out = np.empty_like(arr, dtype=np.int64)
n_samples = np.ceil(arr, out, casting='unsafe')
n_tot = np.sum(n_samples) # corrected due to thresholding

X_dataset = []
U_dataset = []
for j in range(n_dataset):
    X_samples, U_samples = [], []
    for i in range(n_center):
        raw_samples = np.random.normal(0, sigma_star[i], (n_samples[i],dim)) + centers[i]
        X_samples.append(np.mod(raw_samples, np.ones((n_samples[i],dim)) ))
        U_samples.append(U(X_samples[-1]))
    X_dataset.append(X_samples)
    U_dataset.append(U_samples)

#### Gradient descent for various number of data ####

h = 0.1
max_iter = 150
accuracy = 0.00001
p0 = np.ones((n_center))/n_center
KL_type = 0

sample_limitations = [100, 200, 400, 800, 1200]
n_limitations = len(sample_limitations)
labels = '$p_f$', '$J_n(p_0)$', '$J_\\infty(p_0)$' #'$J_\\infty(\\langle p_f , \\nu \\rangle)$', 
n_columns = len(labels)
collection = np.zeros((n_limitations, n_dataset, n_columns))

n_quad = 1200
X_quad = (5*(2*np.arange(n_quad)/n_quad - 1)).reshape(-1,1) # integrals computed on [-4, 4]

for j in range(n_dataset):
    print('----<#### Dataset '+str(j)+' ####>----')
    for k in range(len(sample_limitations)):
        ## Data limitation to k ##
        
        n_lim = sample_limitations[k]
        print('** Starting batch for '+str(n_lim)+' datapoints **')
        X_limited, U_samples_limited = [], []
        for i in range(n_center):
            X_limited.append(X_dataset[j][i][:n_lim])
            U_samples_limited.append(U_dataset[j][i][:n_lim])

        ## Training ##
        solver = mode_weights_torus.GD_estimator(KL_type, p0, U_samples_limited, X_limited, h, accuracy, max_iter, ['silverman']*n_center)
        solver.optimise()

        alpha = solver.alpha
        p_final = solver.p_list[-1]
        ## Data collection ##
        batch_collection = np.array([p_final[0], solver.surrogate_function(p0), solver.objective_function(U, nu, X_quad, p0)]) # solver.objective_function_measure(U, X_quad)
        collection[k][j] = batch_collection

file_path = os.path.join(storage_folder, file_name_output + ".npy")
np.save(file_path, collection)
file_path = os.path.join(storage_folder, file_name_output + ".txt")
with open(file_path, "w") as f:
    f.write(repr(sample_limitations) + "\n")
    f.write(repr(labels))