"""
THIS SCRIPT INTAKES THE OPTIMAL PARAMETERS AND N AND GENERATES:
    
    > 1. FULL DATA FILE (MSE) (TXT)
    > 2. MRSQ PLOT + CONFIDENCE (EPS)
    > 3. SIMPLE DATA FILE (MSE) (TXT)
    > 4. GROUND ERROR PLOT (EPS)
    > 5. BAYESIAN PLOT (EPS)
    
NOTE THAT MSE AND BAYESIAN PLOTS ARE BOTH IN THIS FILE BECAUSE THE BAYESIAN 
USES AS PRIOR THE PARAMETERS CALCULATED THROUGH MSE, AS FOR A LINEAR 
REGRESSION THE MAXIMUM LIKELIHOOD PARAMETERS ASSUMING A NORMAL DISTRIBUTION 
CORRESPOND TO THE PARAMETERS CALCULATED BY A MSE APPROACH
    
SOME OPTIMAL PARAMETERS:
    > N=1: [a,b,c]=[0,0,-1.68484848]
    > N=3: [a,b,c]=[-0.06636363,2.05,-2.505]
    > N=8: [a,b,c]=[-0.014,0.56727272,-2.64286]
 
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy import stats

"""
----------------- DATA IMPORT -----------------

NOTE THAT THE DATA FILE SHOULD HAVE SAME STRUCTURE AS DATA.TXT ON GITHUB

https://github.com/mario-b-amaro/multilinear-regression

IMPORTANT: KEEP  DATA.TXT ON THE SAME FOLDER AS THE SCRIPT

"""

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

"""
--------------- DEFINE PARAMETERS ----------------
"""

nn=3
a=-0.06636363
b=2.05
c=-2.505

combo=[a,b,c]

# DETERMINE CLOSEST LINE

dlist=np.identity(nn)
conj_list=[] # List with "coordinates" [a,b,c], data point (i) and which line the point is fitting to (d) and with the square of the error associated with each "coordinate" above

for i in range(len(xlist)):
    for d in range(len(dlist)):
        sse=[] # List for the point's square error associated with each line
        for n in range(nn):
            sse.append(((ylist[i]-n*(a*xlist[i]+b)-c)*dlist[d][n])**2) # Square error calulation
        ssesum=sum(sse) # Choice of the line closest to the point
        conj_list.append([[i,a,b,c,d],ssesum])
        

sumlist=[]
line=[]

for k in range(len(conj_list)):
    if conj_list[k][0][1]==a and conj_list[k][0][2]==b and conj_list[k][0][3]==c and conj_list[k][0][4]==0:
        sumtempn=[]
        for n in range(nn):
            sumtempn.append(conj_list[k+n][1])             
        ddef=min(sumtempn)
        sumlist.append(ddef)
        line.append(sumtempn.index(ddef))

"""
------------ 1. OBTAIN DATA FILE (MSE) (TXT) -----------

"""

k=np.loadtxt('Data.txt', skiprows=1, usecols=(0,1,2,3,4,6,7))
f= open('MSE_Model&Errors_Source.txt',"w+")

modelpred=[line[i]*(combo[0]*xlist[i]+combo[1])+combo[2] for i in range(len(xlist))] # Model prediction for log(|RF(R)|^2)
modelerr_log=[ylist[i]-modelpred[i] for i in range(len(ylist))] # Error_log = log(|RF(R)|^2) - Model_log(|RF(R)|^2)
modelerr=[10**ylist[i]-10**modelpred[i] for i in range(len(ylist))] # Error = |RF(R)|^2 - Model_|RF(R)|^2
test_log=[(y_ulist[i]>10**modelpred[i] and y_llist[i]<10**modelpred[i]) for i in range(len(modelpred))] # Test if |RF(R)|^2_l < Model < |RF(R)|^2_u
up_err=[y_ulist[i]-10**ylist[i] for i in range(len(ylist))] 
low_err=[10**ylist[i]-y_llist[i] for i in range(len(ylist))]

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

"""
------------ 2. MRSQ PLOT + CONFIDENCE (EPS) -----------

"""

sigma=1 # Confidence bands: 0%=0, 50%=0.6745, 68.3%=1, 90%=1.645, 95%=1.960, 99%=2.576

lineerrs=[]
for n in range(nn):
    lineerrs.append([])

for i in range(len(modelerr_log)):
    lineerrs[line[i]].append(float(modelerr_log[i])**2)
    
errsperlin=[]
for lin in range(len(lineerrs)):
    errsperlin.append(m.sqrt(sum(lineerrs[lin])/(len(lineerrs[lin])+1e-100)))

xx=np.linspace(15,27,10)

plt.rcParams.update({'font.size': 17}) # Adjust font size (10 is default, 15 is fine, 20 is big, 30 is very big, etc.)
for n in range(nn):
    plt.fill_between(xx,n*(combo[0]*xx+combo[1])+combo[2]-sigma*errsperlin[n],n*(combo[0]*xx+combo[1])+combo[2]+sigma*errsperlin[n],alpha=0.2,color='lightgreen')
    plt.plot(xx,n*(combo[0]*xx+combo[1])+combo[2],'--k')
plt.scatter(xlist,ylist,s=75,c='k',marker='.')
plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
plt.ylim([-2.85,-0.5])
plt.xlim([16,24])
plt.xlabel(chr(961)+'´')
plt.ylabel('$log_{10}(|RF(R)|^2)$')
plt.tight_layout()
plt.savefig('MSE_Plot.eps', format='eps')
plt.show()  

"""
------------ 3. PLOT SOURCE FILE (MSE) (TXT) -----------

"""

f= open('MSE_Plots_Source.txt',"w+")

f.write('A Z rho´ log(|RF(R)|^2) \n')
for i in range(len(k)):
    f.write(str(int(k[i][0]))+' '+str(int(k[i][1]))+' '+str(xlist[i])+' '+str(ylist[i])+'\n')

f.close()

"""
------------ 4. GROUND ERROR PLOT (EPS) ---------------

"""

stdv=(sum([abs(float(modelerr[i])) for i in range(len(modelerr))])/len(modelerr))

xrange=np.linspace(0,len(modelerr),100)
plt.figure(constrained_layout=True)
plt.plot(xrange,0*xrange,'--r')
plt.fill_between(xrange,0*xrange+stdv,0*xrange-stdv*sigma,alpha=0.2,color='lightcoral')
plt.scatter(range(len(modelerr)),[float(modelerr[i]) for i in range(len(modelerr))],s=75,c='k',marker='.')
plt.errorbar(range(len(modelerr)),[float(modelerr[i]) for i in range(len(modelerr))],yerr=[low_err,up_err],c='k',fmt='.')
plt.ylabel(r'$\mathregular{|RF(R)|^2-10^{(k-1)(\alpha x_{i}+\beta)-o}}$')
plt.xlabel('Datapoint (i)')
plt.tight_layout()
plt.savefig('abserrors.eps', format='eps', dpi=800)
plt.show()
    
#plt.plot(err_sum_list,'k')
plt.show()

"""
------------ 5. BAYESIAN PLOT (EPS) -----------------

Bayesian approach directly adapted from Martin Krasser's code:
https://github.com/krasserm/bayesian-machine-learning/tree/dev/bayesian-linear-regression

Refer to it for a complete documentation regarding the process

"""

# Collapsed Linear Regression

collapsed=[]
for i in range(len(ylist)):
    collapsed.append(ylist[i]-(line[i]-1)*(xlist[i]*combo[0]+combo[1]))

# Bayesian Functions

def posterior(Phi, t, alpha, beta, return_inverse=False):
    
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N

def posterior_predictive(Phi_test, m_N, S_N, beta):
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    
    return y, y_var

def identity_basis_function(x):
    return x

def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)

def plot_data(x, t):
    plt.scatter(x, t, marker='o', c="k", s=20)


def plot_truth(x, y, label='Truth'):
    plt.plot(x, y, 'k--', label=label)


def plot_predictive(x, y, std, y_label='Prediction', std_label='Uncertainty', plot_xy_labels=True):
    y = y.ravel()
    std = std.ravel()

    plt.plot(x, y, label=y_label,color='k')
    plt.fill_between(x.ravel(), y + std, y - std, alpha = 0.1, label=std_label,color='lightblue')

    if plot_xy_labels:
        plt.xlabel(chr(961)+'´')
        plt.ylabel('$log_{10}(|RF(R)|^2)$')
        
def plot_posterior(mean, cov, w0, w1, m_N):
    resolution = 1000

    x_min=w0-0.6
    x_max=w0+0.6
    y_min=w1-1.1*abs(w1)
    y_max=w1+1.2*abs(w1)
    
    grid_x = np.linspace(x_min, x_max, resolution)
    grid_y = np.linspace(y_min, y_max, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y))
    densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov)
    plt.imshow(densities, cmap='coolwarm', origin='lower', extent=(x_min, x_max, y_min, y_max),aspect='auto')
    plt.scatter(w0, w1, marker='x', c="k", s=45, label='Prior')
    plt.scatter(m_N[0], m_N[1], marker='o', c="k", s=45, label='Predictive')

    plt.xlabel(r'$\mathregular{\alpha}$')
    plt.ylabel(r'$\mathregular{\beta}$+o')

# Bayesian Parameters

f_w0 = combo[1]+combo[2] # Use as prior the "collapsed" MSE parameters
f_w1 = combo[0]

beta = 1/(sum(sumlist)/(len(xlist))) # beta=1/variance
alpha = (sum(sumlist)/(len(xlist))) # alpha=1

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

X_test = np.linspace(16, 25, 100).reshape(-1,1)
    
Phi_test = expand(X_test, identity_basis_function) # Design matrix of test observations

plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.4)

for i, N in enumerate([len(collapsed)]):
    X_N = X[:N]
    t_N = t[:N]

    Phi_N = expand(X_N, identity_basis_function) # Design matrix of training observations
    
    m_N, S_N = posterior(Phi_N, t_N, alpha, beta) # Mean and covariance matrix of posterior
    
    y, y_var = posterior_predictive(Phi_test, m_N, S_N, beta) # Mean and variances of posterior predictive 

    plt.rcParams.update({'font.size': 35})
    plt.subplot()
    plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    plot_posterior(m_N, S_N, f_w0, f_w1, m_N)
    #plt.legend()
    plt.tight_layout()
    plt.savefig('PosteriorPlot.eps', format='eps')
    plt.show()

    plt.rcParams.update({'font.size': 15})
    plt.subplot()
    for n in range(nn):
        plot_truth(X_test, n*(f_w1*X_test+f_w0)-(n-1)*combo[2], label=None)
    for n in range(nn):
        plot_predictive(X_test, n*y-(n-1)*combo[2], np.sqrt(y_var))
    plt.ylim([-2.85,-0.5])
    plt.xlim([16,24])
    plot_data(xlist, ylist)
    plt.tight_layout()
    plt.show()
    
    plt.savefig('BayesianPlot.eps', format='eps')
