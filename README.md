# Variational inference for multimodal Gibbs measure, the code

## Algorithms

In the folder 'algorithms', one can find implementations of the two algorithms studied during this internship. It is composed of the files:

- mode_weights.py
- mode_weights_torus.py
- particle_weights.py
- better_mode_weights.py

Each one of these files define a class named 'GD_estimator'. It can be initialised with the known datapoints, the gradient descent parameters, and the rule used to determine the bandwidth for the sampled density kernel density estimation (KDE). Then, the method 'optimise' can be used to run the gradient descent. Finally, it is possible to evaluate the resulting estimator with the method 'evaluate'.

At the bottom of these files, there is a commented code that allows to test these classes.

### mode_weights.py

Having data $\left(x_{j,i} \in \mathbb{R}^d, U(x_{j,i}) \in \mathbb{R}\cup\{+\infty\}\right)_{i \in \{1,\ldots, K\}, j \in \{1,\ldots,n_i\}}$ it implements the gradient descent on:

$$J_n : p \in \Delta_{K-1} \mapsto \frac{1}{K}\sum_{i=1}^{K} \frac{1}{n_i} \sum_{j=1}^{n_i} \frac{f_{x_{j,i}}(\langle p , \hat{\nu}(x_{j,i})\rangle)}{\hat{\nu}_i(x_{j,i})}$$

$f_x$ being either $e^{-U(x)}\log(.)$ or $.(U(x) + \log(.))$ and $\hat{\nu} = (\frac{1}{n_i}\sum_{j=1}^{n_i} g_{\alpha_i}(.-x_{j,i}))_{i \in \{1,\ldots,K\}}$ where $g$ is the Gaussian kernel. Here, $(\alpha_i)_{i\in \{1,\ldots,K\}}$ are determined by selection rules among the implemented ones. We denote $p_f \in \Delta_{K-1}$ the gradient descent outputted weights, the density estimator becomes $\langle p_f, \hat{\nu}(.)\rangle$.

### mode_weights_torus.py

This is essentially the same algorithm as in mode_weights.py but for unit torus valued measures. Having data $\left(x_{j,i} \in [0,1]^d, U(x_{j,i}) \in \mathbb{R}\cup\{+\infty\}\right)_{i \in \{1,\ldots, K\}, j \in \{1,\ldots,n_i\}}$ it implements the gradient descent on:

$$J_n : p \in \Delta_{K-1} \mapsto \frac{1}{K}\sum_{i=1}^{K} \frac{1}{n_i} \sum_{j=1}^{n_i} \frac{f_{x_{j,i}}(\langle p , \hat{\nu}(x_{j,i})\rangle)}{\hat{\nu}_i(x_{j,i})}$$

Here, $\hat{\nu}$ becomes: $\hat{\nu} = (\frac{1}{n_i}\sum_{j=1}^{n_i} \sum_{q \in Q\subset \Z^d}g_{\alpha_i}(.-x_{j,i} + q))_{i \in \{1,\ldots,K\}}$ where $Q = \{-k, \ldots k\}^d$, $k \in \N$. $(\alpha_i)_{i\in \{1,\ldots,K\}}$ are determined by selection rules among the implemented ones. We denote $p_f \in \Delta_{K-1}$ the gradient descent outputted weights, the density estimator is $\langle p_f, \hat{\nu}(.)\rangle$. 

### particle_weights.py

Having data $\left(x_i \in \mathbb{R}^d, U(x_i) \in \mathbb{R}\cup\{+\infty\}\right)_{i \in \{1,\ldots, n\}}$ it implements the gradient descent on:

$$J_n: p\in \Delta_{n-1} \mapsto \frac{1}{n} \sum_{i=1}^n f_{x_i}\left(\sum_{j=1}^n p_j g_{\alpha}(x_i - x_j) \right) \frac{1}{\hat{\nu}(x_i)} + \sum_{k=1}^m \lambda_k R_k(p, x)$$

With $\hat{\nu} = \frac{1}{n}\sum_{j=1}^n g_{\alpha}(. - x_j)$ and $\alpha > 0$ determined by a bandwidth selection rule among the implemented ones. The other additive terms play the role of regularisers. $(\lambda_k)_{k \in \{1,\ldots, m\}}$ are positive multiplicative constants and $(p,x) \mapsto (R_k(p, x))_{k \in \{1,\ldots, m\}}$ are regularising functions, such as:

$$R_1(p,x) = n\sum_{i=1}^n p_i^2 $$
$$R_2(p,x) = \sum_{i,j=1}^n \frac{(p_i - p_j)^2}{\max(a_n^2, \|x_i - x_j\|^2)}\frac{1_{\|x_i - x_j \| \leq b_n}}{|B(0, b_n)|}$$

 We denote $p_f \in \Delta_{K-1}$ the gradient descent outputted weights, the density estimator is $\sum_{j=1}^n p_j g_{\alpha}(. - x_j)$

 ### better_mode_weights.py

 Having data $\left(x_{j,i} \in \mathbb{R}^d, U(x_{j,i}) \in \mathbb{R}\cup\{+\infty\}\right)_{i \in \{1,\ldots, K\}, j \in \{1,\ldots,n_i\}}$ it implements the gradient descent on:

$$J_n : p \in \Delta_{K-1} \mapsto \sum_{i=1}^{K} \frac{p_i}{n_i} \sum_{j=1}^{n_i} \left[ U + \log(\langle p ,\hat{\nu} \rangle) \right](x_{j,i})$$

With $\hat{\nu} = (\frac{1}{n_i}\sum_{j=1}^{n_i} g_{\alpha_i}(.-x_{j,i}))_{i \in \{1,\ldots,K\}}$ where $g$ is the Gaussian kernel. Here, $(\alpha_i)_{i\in \{1,\ldots,K\}}$ are determined by selection rules among the implemented ones. We denote $p_f \in \Delta_{K-1}$ the gradient descent outputted weights, the density estimator becomes $\langle p_f, \hat{\nu}(.)\rangle$.

 ## Report figures and tables

 The code used to generate the report figures and tables for the first algorithm study consists in the notebook report_data.ipynb and the file from_data_to_latex_array.py. In the notebook, calculations are performed, then stored in the folder table_storage. One can then use the Python file to compute averages and standard deviations on the data and output the resulting Latex table.

 For the second algorithm study, some tests are available in the notebook particle_tests.ipynb