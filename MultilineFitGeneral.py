# THIS SCRIPT IS FITTING log(|RF(R)|^2). ADAPT AS NEEDED

import numpy as np
import math as m
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

alist=np.linspace(0.086,0.093,11)
blist=np.linspace(-2.85,-2.77,11)
clist=np.linspace(-1.25,-1.38,11)

nn=2 # Number of lines to fit the data to
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

# ------ PLOTTING THE DATA WITH THE FIT ------

xx=np.linspace(17.3,23.7,10)

plt.scatter(xlist,ylist)
plt.ylim([-2.8,-0.6])
plt.xlim([17,24])
for n in range(nn):
    plt.plot(xx,n*(combo[0]*xx+combo[1])+combo[2],'--k')
plt.show()    
    
plt.plot(err_sum_list,'k')

# ------ CALCULATING ERROR FILE ------

k=np.loadtxt('Data.txt', skiprows=1, usecols=(0,1,2,3,4,6,7))
f= open('Data_w_Errors_RF(R)2.txt',"w+")

modelpred=[linesopt[i]*(combo[0]*xlist[i]+combo[1])+combo[2] for i in range(len(xlist))] # Model prediction for log(|RF(R)|^2)
modelerr_log=[ylist[i]-modelpred[i] for i in range(len(ylist))] # Error_log = log(|RF(R)|^2) - Model_log(|RF(R)|^2)
modelerr=[10**ylist[i]-10**modelpred[i] for i in range(len(ylist))] # Error = |RF(R)|^2 - Model_|RF(R)|^2
test_log=[(y_ulist[i]>10**modelpred[i] and y_llist[i]<10**modelpred[i]) for i in range(len(modelpred))] # Test if |RF(R)|^2_l < Model < |RF(R)|^2_u
up_err=[y_ulist[i]-10**ylist[i] for i in range(len(ylist))] 
low_err=[10**ylist[i]-y_llist[i] for i in range(len(ylist))]

# ------ OBTAINING GROUND ERROR PLOT ------

xrange=np.linspace(0,len(modelerr),100)
plt.figure(constrained_layout=True)
plt.plot(xrange,0*xrange,'--r')
plt.scatter(range(len(modelerr)),modelerr,s=75,c='k',marker='.')
plt.errorbar(range(len(modelerr)),modelerr,yerr=[low_err,up_err],c='k',fmt='.')
plt.ylabel(r'$\mathregular{|RF(R)|^2-10^{(k-1)(\alpha x_{i}+\beta)-o}}$')
plt.xlabel('Data Point (i)')
plt.savefig('abserrors', dpi=800)

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

f.write('A Z |RF(R)|^2 |RF(R)|^2_u |RF(R)|^2_l F(R)/PDU PDU || Model_log(|RF(R)|^2) Error_log Model_|RF(R)|^2 Error |RF(R)|^2_l < Model < |RF(R)|^2_u \n')

for i in range(len(k)):
    f.write(str(int(k[i][0]))+' '+str(int(k[i][1]))+' '+str(format(k[i][2],'.6f'))+' '+str(format(k[i][3],'.5f'))+' '+str(format(k[i][4],'.5f'))+' '+str(format(k[i][5],'.4f'))+' '+str(format(k[i][6],'.4f'))+' || '+str(format(modelpred[i],'.4f'))+' '+modelerr_log[i]+' '+str(format(10**modelpred[i],'.4f'))+' '+modelerr[i]+' '+str(test_log[i])+'\n')
    
f.close()

fin=time.time() # UNIX TIME END OF RUN

print('Time to run:', fin-begg)
print('The set of optimal parameters is:',combo)
print('The variance is:',min(err_sum_list)/(len(xlist)),'(n =',nn,')')

