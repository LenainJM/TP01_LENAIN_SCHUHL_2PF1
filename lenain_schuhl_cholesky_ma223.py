#------------------------------------------------
# NOMS 1: LENAIN Jean-Marc
# NOMS 2: SCHUHL Thomas
# CLASSE: 2PF1
# TP_Cholesky_Ma223
#------------------------------------------------
#-------------------IMPORT-----------------------
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
#------------------------------------------------
#------------------------------------------------
#----PARTIE_1:
#--QUESTION_1:

A=np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
A = B = np.array(A,dtype=float)
B=np.array([[6,-9,-7]])
B = np.array(B,dtype=float)

def cholesky(A):
    n, m = np.shape(A)
    L = np.zeros_like(A)
    A=np.array(A,float)
    for i in range (n):
        for j in range (i,n):
            if (j == i):
                L[j,i]=np.sqrt(A[j,i]-np.sum(L[j,:i]**2))
            else:
                L[j, i] = (A[j, i] - np.sum(L[j, :i]*L[i, :i])) / L[i, i]
    
    Lt=np.transpose(L)
    print(Lt)
    return L


#------------------------------------------------
#------------------------------------------------
#----PARTIE_2:
#--QUESTION_1:

def ResolCholesky(A, B):
    L = cholesky(A)
    U = np.transpose(L)
    n, m = np.shape(L)
    y = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        y[i] = (B[i] - np.sum(np.dot(L[i, :i], y[:i]))) / L[i, i]
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(np.dot(U[i, i+1:n], x[i+1:n]))) / U[i, i]
    x = np.reshape(x, (n, 1))
    return x

C=cholesky(A)
print(C)


#------------------------------------------------
#------------------------------------------------
#----PARTIE_3:
#--QUESTION_1:

#ResolCholesky:

ListeNcholesky = []
tcholesky= []
echolesky= []
nb_logcholesky= []
temps_logcholesky= []
nbrcholesky = len(ListeNcholesky)
Tempscholesky=np.zeros(nbrcholesky)

for i in range (1,1000,5):
    
    m = np.random.rand(i,i)
    A = np.dot(m, np.transpose(m))
    B = np.random.rand(i,1)
    starttcholesky = time.perf_counter()
    X=ResolCholesky(A,B)
    stoptcholesky = time.perf_counter()
    
    tcholesky.append(stoptcholesky-starttcholesky)
    nb_logcholesky.append(math.log(i))
    ListeNcholesky.append(i)
    temps_logcholesky.append(math.log(stoptcholesky-starttcholesky))
    erreurcholesky = (np.linalg.norm(np.dot(A,X)-B))
    echolesky.append(erreurcholesky)


plt.plot(ListeNcholesky,tcholesky)
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()

#--QUESTION_2:

plt.plot(ListeNcholesky,echolesky)
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

"""plt.plot(nb_logcholesky,temps_logcholesky)
plt.title("vitesse d'execution de la matrice")
plt.xlabel("log n")
plt.ylabel("log t")
plt.show()"""

#------------------------------------------------
#------------------------------------------------
#---- COMPARAISON ----

##################################--GAUSS--##################################

Aaug=np.array([[2,5,6,7],[4,11,9,12],[-2,-8,7,3]])
print("La matrice augmentée:\n",Aaug)

def ReductionGauss(Aaug):
    n,m=np.shape(Aaug)   
    for i in range (0,n-1):
        for j in range (i+1,n):
            if Aaug[i,i]==0:              
                copy=Aaug[j,:].copy()
                Aaug[j,:]=Aaug[i,:]
                Aaug[i,:]=copy
            else:                
                gik = Aaug[j,i]/Aaug[i,i]      
                Aaug[j,:]=Aaug[j,:]-gik*Aaug[i,:]    
    return Aaug
print("")
print("La reduction de Gauss (triangle supérieur):\n",ReductionGauss(Aaug))
print("")

A=np.array([[1,1,1,1,1],[0,2,-5,0,-1],[0,0,1,-2,3],[0,0,0,4,-4]])
print(A)
def ResolutionSystTriSup(A):
    n,m=np.shape(A)
    if m !=n+1:
        print ("ce n'est pas une matrice augmentée")
        return 
    x = np.zeros(n)             
    for i in range (n-1,-1,-1):      
        add = 0
        for k in range (i,n):
            add = add+x[k]*A[i,k]
        x[i]=(A[i,n]-add)/A[i,i]     
    return x        
print("Les valeurs de x,y,z,t sont:\n",ResolutionSystTriSup(A))
print("")

A = np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
B = np.array([[6,-9,-7]])

A=np.array([[1,1,1,1],[2,4,-3,2],[-1,-1,0,-3],[1,-1,4,9]])
B=np.array([[1],
            [1],
            [2],
            [-8]])
def Gauss(A,B):
    Aaug = np.column_stack((A,B))
    Taug = ReductionGauss(Aaug)
    x = ResolutionSystTriSup(Taug)
    print(Aaug)
    return x


def generematrice(n):
    A = np.random.randint (1,100,size=(n,n))
    A=np.array(A,dtype=float)
    det = np.linalg.det(A)
    while det == 0 :
        A= np.random.randint(1,100,size=(n,n))
        A=np.array(A,dtype=float)
        det = np.linalg.det(A)
    A1= np.transpose(A)
    A=A1@A
    B = np.random.randint(1,100,size=(1,n))
    B=np.array(B,dtype=float)
    #print(A)
    #print(B)
    return A,B



#--COURBE--
ListeNgauss = []
tgauss= []
egauss= []
nb_loggauss= []
temps_loggauss= []
nbrgauss = len(ListeNgauss)
Tempsgauss=np.zeros(nbrgauss)

for i in range(1,1000,5):
    C,D=generematrice(i)
    ListeNgauss.append(i)
    A = np.random.rand(i,i) 
    B = np.random.rand(i,1)
    starttgauss = time.perf_counter()
    X=Gauss(A,B)
    stoptgauss = time.perf_counter()
    tgauss.append(stoptgauss-starttgauss)
    nb_loggauss.append(math.log(i))      
    temps_loggauss.append(math.log(stoptgauss-starttgauss))
    erreurgauss = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    egauss.append(erreurgauss)



plt.plot(ListeNgauss,tgauss,"r")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()


plt.plot(ListeNgauss,egauss,"r")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

plt.plot(ListeNcholesky,tcholesky)
plt.plot(ListeNgauss,tgauss,"r")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()

plt.plot(ListeNcholesky,echolesky)
plt.plot(ListeNgauss,egauss,"r")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

"""plt.plot(nb_loggauss,temps_loggauss,"r")
plt.plot(nb_logcholesky,temps_logcholesky)
plt.title("vitesse d'execution de la matrice")
plt.xlabel("log n")
plt.ylabel("log t")
plt.show()"""

##################################--LU--##################################

A = np.array([[3.,2,-1,4],[-3.,-4,4,-2],[6.,2,2,7],[9.,4,2,18]])
B = np.array([[4.],[-5],[-2],[13]])
def DecompositionLU(A):
    U = np.copy(A)
    n,m = U.shape
    L=np.eye(n,n)
    for i in range (n):
        for j in range (i+1,n):
            L[j,i]= U[j,i]/U[i,i]
            U[j,:] -= U[j,i]/U[i,i]*U[i,:]
    return L,U

#-COURBES

def ResolutionLU(L,U,B):
    Y=Gauss(L,B)
    X=Gauss(U,Y)
    return(X)
C=DecompositionLU(A)
D=ResolutionLU(C[0],C[1],B)

ListeNLU = []
tLU= []
eLU= []
nb_logLU= []
temps_logLU= []
nbrLU = len(ListeNLU)
TempsLU=np.zeros(nbrLU)

for i in range(1,1000,5):
    C,D=generematrice(i)
    ListeNLU.append(i)
    A = np.random.rand(i,i) 
    B = np.random.rand(i,1)
    starttLU = time.perf_counter()
    L,U=DecompositionLU(A)
    X=ResolutionLU(L,U,B)
    stoptLU = time.perf_counter()
    tLU.append(stoptLU-starttLU)
    nb_logLU.append(math.log(i))      
    temps_logLU.append(math.log(stoptLU-starttLU))
    erreurLU = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    eLU.append(erreurLU)

plt.plot(ListeNLU,tLU,"g")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()


plt.plot(ListeNLU,eLU,"g")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

plt.plot(ListeNcholesky,tcholesky)
plt.plot(ListeNLU,tLU,"g")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()

plt.plot(ListeNcholesky,echolesky)
plt.plot(ListeNLU,eLU,"g")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()


##################################--NUMPY CHOLESKY--##################################

def NumpyCholesky(A, B):
    L = np.linalg.cholesky(A)
    U = np.transpose(L)
    n, m = np.shape(L)
    y = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        y[i] = (B[i] - np.sum(np.dot(L[i, :i], y[:i]))) / L[i, i]
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(np.dot(U[i, i+1:n], x[i+1:n]))) / U[i, i]
    x = np.reshape(x, (n, 1))

    return x

ListeNnumpy = []
tnumpy= []
enumpy= []
nb_lognumpy= []
temps_lognumpy= []
nbrnumpy = len(ListeNnumpy)
Tempsnumpy=np.zeros(nbrnumpy)

for i in range(1,1000,5):
    m=np.random.rand(i,i)
    ListeNnumpy.append(i)
    A = np.dot(m,np.transpose(m))
    B = np.random.rand(i,1)
    starttnumpy = time.perf_counter()
    X=NumpyCholesky(A,B)
    stoptnumpy = time.perf_counter()
    tnumpy.append(stoptnumpy-starttnumpy)
    nb_lognumpy.append(math.log(i))      
    temps_lognumpy.append(math.log(stoptnumpy-starttnumpy))
    erreurnumpy = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    enumpy.append(erreurnumpy)

plt.plot(ListeNnumpy,tnumpy,"k")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()


plt.plot(ListeNnumpy,enumpy,"k")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

plt.plot(ListeNcholesky,tcholesky)
plt.plot(ListeNnumpy,tnumpy,"k")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()

plt.plot(ListeNcholesky,echolesky)
plt.plot(ListeNnumpy,enumpy,"k")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

##################################--LINALGSOLVE--##################################

ListeNsolve = []
tsolve= []
esolve= []
nb_logsolve= []
temps_logsolve= []
nbrsolve = len(ListeNsolve)
Tempssolve=np.zeros(nbrsolve)

for i in range(1,1000,5):
    A=np.random.rand(i,i)
    ListeNsolve.append(i)
    B = np.random.rand(i,1)
    starttsolve = time.perf_counter()
    X=np.linalg.solve(A,B)
    stoptsolve = time.perf_counter()
    tsolve.append(stoptsolve-starttsolve)
    nb_logsolve.append(math.log(i))      
    temps_logsolve.append(math.log(stoptsolve-starttsolve))
    erreursolve = (np.linalg.norm(np.dot(A,X)-B))
    esolve.append(erreursolve)

plt.plot(ListeNsolve,tsolve,"m")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()


plt.plot(ListeNsolve,esolve,"m")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()

plt.plot(ListeNcholesky,tcholesky)
plt.plot(ListeNsolve,tsolve,"m")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()

plt.plot(ListeNcholesky,echolesky)
plt.plot(ListeNsolve,esolve,"m")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()


#########--COURBES TOTAL--########

plt.plot(ListeNcholesky,tcholesky)
plt.plot(ListeNgauss,tgauss,"r")
plt.plot(ListeNLU,tLU,"g")
plt.plot(ListeNnumpy,tnumpy,"k")
plt.plot(ListeNsolve,tsolve,"m")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul")
plt.show()



plt.plot(ListeNcholesky,echolesky)
plt.plot(ListeNgauss,egauss,"r")
plt.plot(ListeNLU,eLU,"g")
plt.plot(ListeNnumpy,enumpy,"k")
plt.plot(ListeNsolve,esolve,"m")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Incertitude")
plt.show()
