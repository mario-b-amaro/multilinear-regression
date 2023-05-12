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

alist=np.linspace(-0.1,-0.05,10)
blist=np.linspace(2,2.5,10)
clist=np.linspace(-2.35,-2.65,7)

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
max_val_list=[]  
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
            max_val_list.append(max(sumtemp))
            err_sum_list.append(sum(sumtemp))
            coord_sum_list.append([alist[a],blist[b],clist[c]])
            line_list.append(line)

index=err_sum_list.index(min(err_sum_list))
indexminimax=max_val_list.index(min(max_val_list))

combo=coord_sum_list[index] # Combo list is the optimal [a,b,c] combination
combominimax=coord_sum_list[indexminimax]

linesopt=line_list[index]

# ------ PLOTS -------

lineerrs=[]
for n in range(nn):
    lineerrs.append([])
    
errsperlin=[]
for lin in range(len(lineerrs)):
    errsperlin.append(m.sqrt(sum(lineerrs[lin])/(len(lineerrs[lin])+1e-100)))

xx=np.linspace(17.3,23.7,10)

# MSE

plt.scatter(xlist,ylist,s=75,c='g',marker='.')
plt.ylim([-2.8,-0.6])
plt.xlim([17,24])
for n in range(nn):
    plt.plot(xx,n*(combo[0]*xx+combo[1])+combo[2],'--k')
plt.show()    

# MINIMAX

plt.scatter(xlist,ylist,s=75,c='g',marker='.')
plt.ylim([-2.8,-0.6])
plt.xlim([17,24])
for n in range(nn):
    plt.plot(xx,n*(combominimax[0]*xx+combominimax[1])+combominimax[2],'--k')
plt.show()    

# 

fin=time.time() # UNIX TIME END OF RUN

print('Time to run:', fin-begg)
print('The set of optimal parameters is:',combo)
print('The variance is:',min(err_sum_list)/(len(xlist)),'(n =',nn,')')