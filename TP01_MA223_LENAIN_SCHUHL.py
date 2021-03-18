#------------CARTOUCHE-------------------
#NOM1: LENAIN Jean-Marc
#NOM2: SCHUHL Thomas
#Classe: 2PF1
#TP01
#Ma223
#----------------------------------------
#------------IMPORT----------------------
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
import math
#----------------------------------------
#----------------------------------------
#PARTIE_1--------------------------------
#QUESTION_1_1:
print("------Question 1_1------","\n")
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


#QUESTION_1_2:
print("------Question 1_2------","\n")
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


#QUESTION_1_3:
print("------Question 1_3------","\n")
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
print("X,Y,Z,T sont respectivement égaux à:\n",Gauss(A,B))


#QUESTION_1_4:
"""print("------Question 1_4------","\n")
ListeN = []
t= []
e= []
nb_log= []
temps_log= []
nbr = len(ListeN)
Temps=np.zeros(nbr)

for i in range(1,10,5):
    ListeN.append(i)
    A = np.random.rand(i,i) 
    B = np.random.rand(i,1)
    startt = time.perf_counter()
    X=Gauss(A,B)
    stopt = time.perf_counter()
    t.append(stopt-startt)
    nb_log.append(math.log(i))      
    temps_log.append(math.log(stopt-startt))
    erreur = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    e.append(erreur)

print("temps :", stopt-startt,'s')
erreur = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
print ("erreur : ",erreur)

plt.plot(ListeN,t)
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("temps de calcul")
plt.show()

plt.plot(ListeN,e)
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("incertitude")
plt.show()

plt.plot(nb_log,temps_log)
plt.title("vitesse d'execution de la matrice")
plt.xlabel("log n")
plt.ylabel("log t")
plt.show()"""


#----------------------------------------
#----------------------------------------
#PARTIE_2--------------------------------
#QUESTION_2_1:
print("------Question 2_1------","\n")
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


#QUESTION_2_2:
print("------Question 2_2------","\n")
def ResolutionLU(L,U,B):
    Y=Gauss(L,B)
    X=Gauss(U,Y)
    return(X)
C=DecompositionLU(A)
print("La matrice L est donnée par:","\n",C[0],"\n")
print("La matrice U est donnée par:","\n",C[1],"\n")
D=ResolutionLU(C[0],C[1],B)
print("La solution du système AX=B est donnée par X:","\n",D)
print("\n")

"""ListeN = []
t= []
e= []
nb_log= []
temps_log= []
nbr = len(ListeN)
Temps=np.zeros(nbr)

for i in range(1,10,5):
    ListeN.append(i)
    A = np.random.rand(i,i) 
    B = np.random.rand(i,1)
    startt = time.perf_counter()
    L,U=DecompositionLU(A)
    X=ResolutionLU(L,U,B)
    stopt = time.perf_counter()
    t.append(stopt-startt)
    nb_log.append(math.log(i))      
    temps_log.append(math.log(stopt-startt))
    erreur = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    e.append(erreur)

print("temps :", stopt-startt,'s')
erreur = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
print ("erreur : ",erreur)

plt.plot(ListeN,t,"g")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("temps de calcul")
plt.show()

plt.plot(ListeN,e,"g")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("incertitude")
plt.show()

plt.plot(nb_log,temps_log,"g")
plt.title("vitesse d'execution de la matrice")
plt.xlabel("log n")
plt.ylabel("log t")
plt.show()"""


#----------------------------------------
#----------------------------------------
#PARTIE_3--------------------------------
#QUESTION_3_1:
"""A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7, 12, 3]])
def ReductionGaussPartiel(Aaug) :
    n,m=Aaug.shape
    for k in range(m-1):
        for i in range(k,n): 
            if abs(Aaug[k,k]) <= abs(Aaug[i,k]):
                K= np.copy(Aaug)
                Aaug[k,:]=Aaug[i,:]
                Aaug[i,:]=K[k,:]
                print(Aaug)
        for i in range (k+1,n):
            gik=Aaug[i,k]/Aaug[k,k]
            Aaug[i,:]=Aaug[i,:]-gik*Aaug[k,:]
    return Aaug

def GaussChoixPivotPartiel(A,B):
    Aaug = np.concatenate((A, B.T), axis = 1)
    print(Aaug)
    Taug=ReductionGaussPartiel(Aaug)
    X= ResolutionSystTriSup(Taug)
    return(X)
A=GaussChoixPivotPartiel(A,B)
print("A=\n",A)"""

print("------Question 3_1------","\n")
def GaussChoixPivotPartiel(A, B):
    n, m = A.shape
    for i in range(n-1):
        num = i
        for j in range(i, n):
            if abs(A[j, i]) > abs(A[num, i]):
                num = j
            if num != i:
                for k in range(i, n):   
                    temp = A[num, k].copy()
                    A[num, k] = A[i, k]
                    A[i, k] = temp
                temp1 = B[num].copy()
                B[num] = B[i]
                B[i] = temp1
        for j in range(i+1, n):
            g = A[j, i] / A[i, i]
            for k in range(i, n):
                A[j, k] = A[j, k] - g * A[i, k]
            B[j] = B[j] - g * B[i]
    Taug = np.column_stack((A, B))
    X = ResolutionSystTriSup(Taug)
    return X


#QUESTION_3_2:
"""A = np.array([[3,2,-1,4],[-3.,-4,4,-2],[6,2,2,7],[9.,4,2,18]])
B = np.array([[4.],[-5],[-2],[13]])
def GaussChoixPivotTotal(A,B):
    n = A.shape[0]
    if (A.shape == (n,n)):
        if (B.shape == (n,1)):
            A = np.append(A,B,axis = 1)
            print(" Matrice augmentée :\n", A)
            ReductionGauss(A)
            print(" 'La supposé triangulaire supérieure serait: ","\n", A) 
            print(" On trouve les solutions suivantes :", D)
GaussChoixPivotTotal(A,B)"""

print("------Question 3_2------","\n")
def GaussChoixPivotTotal(A,B):
    A = np.array(A, float)
    B = np.array(B, float)
    n,m= A.shape
    ref = np.arange(n)
    for i in range(n):
        for j in range(i, n):
            if abs(A[j, i]) > abs(A[i, i]):
                temp = A[:, j].copy()  
                A[:, j] = A[:, i]
                A[:, i] = temp
                temp1 = ref[j].copy()
                ref[j] = ref[i]
                ref[i] = temp1
    
    X = GaussChoixPivotPartiel(A, B)
    for i in range(n-1, -1, -1):
        for j in range(0, i):
            if ref[j] > ref[j+1]:
                temp = ref[j]
                ref[j] = ref[j+1]
                ref[j+1] = temp
                temp1 = X[j].copy()
                X[j] = X[j+1]
                X[j+1] = temp1

    return X


#----------------------------------------
#----------------------------------------



#------------ZONE DE COURBES-------------
#----GAUSS
ListeNgauss = []
tgauss= []
egauss= []
nb_loggauss= []
temps_loggauss= []
nbrgauss = len(ListeNgauss)
Tempsgauss=np.zeros(nbrgauss)

for i in range(1,20,2):
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


#----LU
ListeNLU = []
tLU= []
eLU= []
nb_logLU= []
temps_logLU= []
nbrLU = len(ListeNLU)
TempsLU=np.zeros(nbrLU)

for i in range(1,20,2):
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
    
#----PivotPartiel

ListeNPP = []
tPP= []
ePP= []
nb_logPP= []
temps_logPP= []
nbrPP = len(ListeNPP)
TempsPP=np.zeros(nbrPP)
for i in range(1,20,2):
    ListeNPP.append(i)
    A = np.random.rand(i,i) 
    B = np.random.rand(i,1)
    starttPP = time.perf_counter()
    X=GaussChoixPivotPartiel(A,B) #choix de la fonction pour les courbes
    stoptPP = time.perf_counter()
    tPP.append(stoptPP-starttPP)
    nb_logPP.append(math.log(i))      
    temps_logPP.append(math.log(stoptPP-starttPP))
    erreurPP = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    ePP.append(erreurPP)
    
#----PivotTotal

ListeNPT = []
tPT= []
ePT= []
nb_logPT= []
temps_logPT= []
nbrPT = len(ListeNPT)
TempsPT=np.zeros(nbrPT)
for i in range(1,20,2):
    ListeNPT.append(i)
    A = np.random.rand(i,i) 
    B = np.random.rand(i,1)
    starttPT = time.perf_counter()
    X=GaussChoixPivotTotal(A,B) #choix de la fonction pour les courbes
    stoptPT = time.perf_counter()
    tPT.append(stoptPT-starttPT)
    nb_logPT.append(math.log(i))      
    temps_logPT.append(math.log(stoptPT-starttPT))
    erreurPT = (np.linalg.norm(np.dot(A,X)-np.ravel(B)))
    ePT.append(erreurPT)


#---COURBES----
"""plt.plot(ListeNgauss,tgauss)
plt.plot(ListeNLU,tLU,"r")
plt.plot(ListeNPP,tPP,"g")
plt.plot(ListeNPT,tPT,"k")
plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("temps de calcul")
plt.show()

plt.plot(ListeNgauss,egauss)
plt.plot(ListeNLU,eLU,"r")
plt.plot(ListeNPP,ePP,"g")
plt.plot(ListeNPT,ePT,"k")
plt.title("Taux d'erreur en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("incertitude")
plt.show()

plt.plot(nb_loggauss,temps_loggauss)
plt.plot(nb_logLU,temps_logLU,"r")
plt.plot(nb_logPP,temps_logPP,"g")
plt.plot(nb_logPT,temps_logPT,"k")
plt.title("vitesse d'execution de la matrice")
plt.xlabel("log n")
plt.ylabel("log t")
plt.show()"""
