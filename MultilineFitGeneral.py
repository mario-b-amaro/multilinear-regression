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

alist=np.linspace(-0.014,-0.01,1)
blist=np.linspace(0.56727272,0.75,1)
clist=np.linspace(-2.64286,-2.58,1)

nn=8 # Number of lines to fit the data to
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

plt.rcParams.update({'font.size': 10}) # Adjust font size (10 is default, 15 is fine, 20 is big, 30 is very big, etc.)
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

"""
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
"""
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
    plt.scatter(x, t, marker='o', c="k", s=50)


def plot_truth(x, y, label='Truth'):
    plt.plot(x, y, 'k--', label=label)


def plot_predictive(x, y, std, y_label='Prediction', std_label='Uncertainty', plot_xy_labels=True):
    y = y.ravel()
    std = std.ravel()

    plt.plot(x, y, label=y_label)
    plt.fill_between(x.ravel(), y + std, y - std, alpha = 0.1, label=std_label)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')

# Bayesian Parameters

f_w0 = combo[1]+combo[2] # Use as prior the "collapsed" MSE parameters
f_w1 = combo[0]

beta = 1/(min(err_sum_list)/(len(xlist))) # beta=1/variance
alpha = 1 # alpha=1

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
    
Phi_test = expand(X_test, identity_basis_function) # Design matrix of test observations

plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.4)

for i, N in enumerate([len(collapsed)]):
    X_N = X[:N]
    t_N = t[:N]

    Phi_N = expand(X_N, identity_basis_function) # Design matrix of training observations
    
    m_N, S_N = posterior(Phi_N, t_N, alpha, beta) # Mean and covariance matrix of posterior
    
    y, y_var = posterior_predictive(Phi_test, m_N, S_N, beta) # Mean and variances of posterior predictive 

    plt.subplot()
    plot_truth(X_test, f_w1*X_test+f_w0, label=None)
    plot_predictive(X_test, y, np.sqrt(y_var))
    plt.ylim([min(collapsed)-0.3*(max(collapsed)-min(collapsed)),max(collapsed)+0.4*(max(collapsed)-min(collapsed))])
    plot_data(X_N, t_N)
    plt.legend(prop={'size': 25})
    
plt.savefig('BayesianPlot.eps', format='eps')


# ------- END ----------

fin=time.time() # UNIX TIME END OF RUN

print('Time to run:', fin-begg)
print('The set of optimal parameters is:',combo)
print('The variance is:',min(err_sum_list)/(len(xlist)),'(n =',nn,')')
