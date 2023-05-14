import numpy as np
import math as m
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

begg=time.time() # UNIX TIME BEGINNING OF RUN

# ------ DATA IMPORT ------

k=np.loadtxt('Data.txt', skiprows=1, usecols=(0,1,2,3,4))

xlist=[] # List for x_i (rho')
ylist=[] # List for y_i (log(|RF(R)|^2))
y_ulist=[] # List for y_i upper bound
y_llist=[] # List for y_i lower bound

for i in range(len(k)):
    r=m.sqrt(k[i][1]*(k[i][0]**(1/3)+1)) 
    xlist.append(r)
    ylist.append(m.log10(k[i][2]))
    y_ulist.append(k[i][3])
    y_llist.append(k[i][4])

# ------ DEFINING N AND TEST INTERVAL ------

# Define an interval and step for each parameter a, b, c

alist=np.linspace(-0.0663636363636,0,1)
blist=np.linspace(2.05,0,1)
clist=np.linspace(-2.505,-0.5,1)

nn=3 # Number of lines to fit the data to
dlist=np.identity(nn) # Identity matrix (delta)

# ------ CALCULATING OPTIMAL PARAMETERS ------

conj_list=[] # List with "coordinates" [a,b,c], data point (i) and which line the point is fitting to (d) and with the square of the error associated with each "coordinate" above

for i in range(len(xlist)):
    for a in alist:
        for b in blist:
            for c in clist:
                for d in range(len(dlist)):
                    sse=[] # List for the point's square error associated with each line
                    for n in range(nn):
                        sse.append(((ylist[i]-n*(a*xlist[i]+b)-c)*dlist[d][n])**2) # Square error calulation
                    ssesum=sum(sse) # Choice of the line closest to the point
                    conj_list.append([[i,a,b,c,d],ssesum])
                    
# Note: for each combination of parameters [a,b,c], each point has one unique d value that tells us which is the closest line. This is the line the point will be contributing to
            
err_sum_list=[]    
coord_sum_list=[]
conj_sum_list=[]
line_list=[]

for a in range(len(alist)):
    for b in range(len(blist)):
        for c in range(len(clist)):
            sumtemp=[]
            line=[]
            for k in range(len(conj_list)):
                if conj_list[k][0][1]==alist[a] and conj_list[k][0][2]==blist[b] and conj_list[k][0][3]==clist[c] and conj_list[k][0][4]==0:
                    sumtempn=[]
                    for n in range(nn):
                        sumtempn.append(conj_list[k+n][1])             
                    ddef=min(sumtempn)
                    sumtemp.append(ddef)
                    line.append(sumtempn.index(ddef))

            conj_sum_list.append([[alist[a],blist[b],clist[c]],sum(sumtemp)]) 
            err_sum_list.append(sum(sumtemp))
            coord_sum_list.append([alist[a],blist[b],clist[c]])
            line_list.append(line)
            
index=err_sum_list.index(min(err_sum_list))

combo=coord_sum_list[index] # Combo list is the optimal [a,b,c] combination
linesopt=line_list[index]

# ------ CALCULATING ERROR FILE ------

k=np.loadtxt('Data.txt', skiprows=1, usecols=(0,1,2,3,4,6,7))
f= open('Data_w_Errors_RF(R)2.txt',"w+")

modelpred=[linesopt[i]*(combo[0]*xlist[i]+combo[1])+combo[2] for i in range(len(xlist))] # Model prediction for log(|RF(R)|^2)
modelerr_log=[ylist[i]-modelpred[i] for i in range(len(ylist))] # Error_log = log(|RF(R)|^2) - Model_log(|RF(R)|^2)
modelerr=[10**ylist[i]-10**modelpred[i] for i in range(len(ylist))] # Error = |RF(R)|^2 - Model_|RF(R)|^2
test_log=[(y_ulist[i]>10**modelpred[i] and y_llist[i]<10**modelpred[i]) for i in range(len(modelpred))] # Test if |RF(R)|^2_l < Model < |RF(R)|^2_u
up_err=[y_ulist[i]-10**ylist[i] for i in range(len(ylist))] 
low_err=[10**ylist[i]-y_llist[i] for i in range(len(ylist))]


# ------ PLOTTING THE DATA WITH THE FIT ------

lineerrs=[]
for n in range(nn):
    lineerrs.append([])

for i in range(len(modelerr_log)):
    lineerrs[linesopt[i]].append(float(modelerr_log[i])**2)
    
errsperlin=[]
for lin in range(len(lineerrs)):
    errsperlin.append(m.sqrt(sum(lineerrs[lin])/(len(lineerrs[lin])+1e-100)))

xx=np.linspace(15,27,10)

for n in range(nn):
    plt.fill_between(xx,n*(combo[0]*xx+combo[1])+combo[2]-errsperlin[n],n*(combo[0]*xx+combo[1])+combo[2]+errsperlin[n],alpha=0.2,color='lightgreen')
    plt.plot(xx,n*(combo[0]*xx+combo[1])+combo[2],'--k')
plt.scatter(xlist,ylist,s=75,c='k',marker='.')
plt.ylim([-2.9,-0.5])
plt.xlim([16,26])
plt.xlabel(chr(961)+'´')
plt.ylabel('$log_{10}(|RF(R)|^2)$')
plt.savefig(r'C:\Users\Mário Amaro\Desktop\PlotsChong\n3_mse.eps', format='eps')
plt.show()  

plt.scatter(xlist,ylist,s=75,c='k',marker='.')
plt.ylim([-2.9,-0.5])
plt.xlim([16,26])
plt.xlabel(chr(961)+'´')
plt.ylabel('$log_{10}(|RF(R)|^2)$')
for n in range(nn):
    plt.plot(xx,n*(combo[0]*xx+combo[1])+combo[2],'--k')
plt.show()    


# ------ OBTAINING GROUND ERROR PLOT ------

xrange=np.linspace(0,len(modelerr),100)
plt.figure(constrained_layout=True)
plt.plot(xrange,0*xrange,'--r')
plt.scatter(range(len(modelerr)),modelerr,s=75,c='k',marker='.')
plt.errorbar(range(len(modelerr)),modelerr,yerr=[low_err,up_err],c='k',fmt='.')
plt.ylabel(r'$\mathregular{|RF(R)|^2-10^{(k-1)(\alpha x_{i}+\beta)-o}}$')
plt.xlabel('Data Point (i)')
plt.savefig('abserrors', dpi=800)
plt.show()
    
#plt.plot(err_sum_list,'k')
plt.show()

# ------ FORMATTING ERROR FILE ------

for i in range(len(test_log)):
    if test_log[i]==True:
        test_log[i]=test_log[i]
    else:
        test_log[i]='x'
        
for i in range(len(modelerr_log)):
    if modelerr_log[i]>0:
        modelerr_log[i]=str(format(modelerr_log[i],'.5f'))
    else:
        modelerr_log[i]=str(format(modelerr_log[i],'.4f'))
        
for i in range(len(modelerr)):
    if modelerr[i]>0:
        modelerr[i]=str(format(modelerr[i],'.5f'))
    else:
        modelerr[i]=str(format(modelerr[i],'.4f'))

f.write('A Z |RF(R)|^2 |RF(R)|^2_u |RF(R)|^2_l F(R)/PDU PDU || Model_log(|RF(R)|^2) Error_log Model_|RF(R)|^2 Error |RF(R)|^2_l<Model<|RF(R)|^2_u \n')

for i in range(len(k)):
    f.write(str(int(k[i][0]))+' '+str(int(k[i][1]))+' '+str(format(k[i][2],'.6f'))+' '+str(format(k[i][3],'.5f'))+' '+str(format(k[i][4],'.5f'))+' '+str(format(k[i][5],'.4f'))+' '+str(format(k[i][6],'.4f'))+' || '+str(format(modelpred[i],'.4f'))+' '+modelerr_log[i]+' '+str(format(10**modelpred[i],'.4f'))+' '+modelerr[i]+' '+str(test_log[i])+'\n')
    
f.close()

# ------ NORMAL DISTRIBUTION CALCULATION ----

# Generate some data for this demonstration.
data = [float(modelerr_log[i]) for i in range(len(modelerr_log))]

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=8, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()


# ------ COLLAPSED LINEAR REGRESSION PLOT ------

collapsed=[]
for i in range(len(ylist)):
    collapsed.append(ylist[i]-(linesopt[i]-1)*(xlist[i]*combo[0]+combo[1]))

plt.scatter(xlist,ylist,s=75,c='r',marker='x')        
plt.scatter(xlist,collapsed,s=75,c='k',marker='.')
plt.plot(xx,xx*combo[0]+combo[1]+combo[2],'--k')




# ------ BAYESIAN FIT --------


def posterior(Phi, t, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N


def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    
    return y, y_var

"""### Example datasets

The datasets used in the following examples are based on $N$ scalar observations $x_{i = 1,\ldots,N}$ which are combined into a $N \times 1$ matrix $\mathbf{X}$. Target values $\mathbf{t}$ are generated from $\mathbf{X}$ with functions `f` and `g` which also generate random noise whose variance can be specified with the `noise_variance` parameter. We will use `f` for generating noisy samples from a straight line and `g` for generating noisy samples from a sinusoidal function.
"""

#f_w0 = -0.3
#f_w1 =  0.5

f_w0 = combo[1]+combo[2]
f_w1 = combo[0]




"""### Basis functions

For straight line fitting, a model that is linear in its input variable $x$ is sufficient. Hence, we don't need to transform $x$ with a basis function which is equivalent to using an `identity_basis_function`. For fitting a linear model to a sinusoidal dataset we transform input $x$ with `gaussian_basis_function` and later with `polynomial_basis_function`. These non-linear basis functions are necessary to model the non-linear relationship between input $x$ and target $t$. The design matrix $\boldsymbol\Phi$ can be computed from observations $\mathbf{X}$ and a parametric basis function with function `expand`. This function also prepends a column vector $\mathbf{1}$ according to $\phi_0(x) = 1$.
"""

def identity_basis_function(x):
    return x


def gaussian_basis_function(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def polynomial_basis_function(x, power):
    return x ** power


def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)

"""### Straight line fitting

For straight line fitting, we use a linear regression model of the form $y(x, \mathbf{w}) = w_0 + w_1 x$ and do Bayesian inference for model parameters $\mathbf{w}$. Predictions are made with the posterior predictive distribution. Since this model has only two parameters, $w_0$ and $w_1$, we can visualize the posterior density in 2D which is done in the first column of the following output. Rows use an increasing number of training data from a training dataset.
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def plot_data(x, t):
    plt.scatter(x, t, marker='o', c="k", s=20)


def plot_truth(x, y, label='Truth'):
    plt.plot(x, y, 'k--', label=label)


def plot_predictive(x, y, std, y_label='Prediction', std_label='Uncertainty', plot_xy_labels=True):
    y = y.ravel()
    std = std.ravel()

    plt.plot(x, y, label=y_label)
    plt.fill_between(x.ravel(), y + std, y - std, alpha = 0.5, label=std_label)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior_samples(x, ys, plot_xy_labels=True):
    plt.plot(x, ys[:, 0], 'r-', alpha=0.5, label='Post. samples')
    for i in range(1, ys.shape[1]):
        plt.plot(x, ys[:, i], 'r-', alpha=0.5)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior(mean, cov, w0, w1):
    resolution = 100

    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)
    plt.imshow(densities, origin='lower', extent=(-1, 1, -1, 1))
    plt.scatter(w0, w1, marker='x', c="r", s=20, label='Truth')

    plt.xlabel('w0')
    plt.ylabel('w1')


def print_comparison(title, a, b, a_prefix='np', b_prefix='br'):
    print(title)
    print('-' * len(title))
    print(f'{a_prefix}:', a)
    print(f'{b_prefix}:', b)
    print()

import matplotlib.pyplot as plt
# %matplotlib inline

# Training dataset sizes
N_list = [1, int(len(xlist)/2), int(len(xlist))]


beta = 1/(min(err_sum_list)/(len(xlist)))
alpha = 1

# Training observations in [-1, 1)

xlist_fbay=[]
for i in range(len(xlist)):
    xlist_fbay.append([xlist[i]])

xlist_fbay=np.array(xlist_fbay)

X= xlist_fbay

# Training target values

collapsed_fbay=[]
for i in range(len(collapsed)):
    collapsed_fbay.append([collapsed[i]])
    
collapsed_fbay=np.array(collapsed_fbay)

t = collapsed_fbay

# Test observations

X_test = np.linspace(17, 25, 100).reshape(-1,1)

# Function values without noise 

y_true = f_w1*X_test+f_w0
    
# Design matrix of test observations
Phi_test = expand(X_test, identity_basis_function)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.4)

for i, N in enumerate(N_list):
    X_N = X[:N]
    t_N = t[:N]

    # Design matrix of training observations
    Phi_N = expand(X_N, identity_basis_function)
    
    # Mean and covariance matrix of posterior
    m_N, S_N = posterior(Phi_N, t_N, alpha, beta)
    
    # Mean and variances of posterior predictive 
    y, y_var = posterior_predictive(Phi_test, m_N, S_N, beta)
    
    # Draw 5 random weight samples from posterior and compute y values
    w_samples = np.random.multivariate_normal(m_N.ravel(), S_N, 5).T
    y_samples = Phi_test.dot(w_samples)
    
    plt.subplot(len(N_list), 3, i * 3 + 1)
    plot_posterior(m_N, S_N, f_w0, f_w1)
    plt.title(f'Posterior density (N = {N})')
    plt.legend()

    plt.subplot(len(N_list), 3, i * 3 + 2)
    plot_data(X_N, t_N)
    plot_truth(X_test, y_true)
    plot_posterior_samples(X_test, y_samples)
    plt.ylim(-2.8, -0.5)
    plt.legend()

    plt.subplot(len(N_list), 3, i * 3 + 3)
    plot_data(X_N, t_N)
    plot_truth(X_test, y_true, label=None)
    plot_predictive(X_test, y, np.sqrt(y_var))
    plt.ylim(-2.8, -0.5)
    plt.legend()
plt.savefig('bayesian.png')


# ------- END ----------

fin=time.time() # UNIX TIME END OF RUN

print('Time to run:', fin-begg)
print('The set of optimal parameters is:',combo)
print('The variance is:',min(err_sum_list)/(len(xlist)),'(n =',nn,')')
