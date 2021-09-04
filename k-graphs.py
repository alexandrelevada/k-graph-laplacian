#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

K-graph Laplacians for supervised metric learning

Created on Wed Jul 24 16:59:33 2019

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import umap
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph_shortest_path as sksp
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy import arccos
from numpy import sin
from numpy import ceil
from scipy.linalg import eigh
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

# ISOMAP implementation
def myIsomap(dados, k, d):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()
    D = sksp.graph_shortest_path(A, directed=False)
    n = D.shape[0]
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

    
# Supervised PCA implementation (variation from paper Supervised Principal Component Analysis - Pattern Recognition)
def SupervisedPCA(dados, labels, d):

    dados = dados.T

    m = dados.shape[0]      # number of samples
    n = dados.shape[1]      # number of features

    I = np.eye(n)
    U = np.ones((n, n))
    H = I - (1/n)*U

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                L[i, j] = 1

    Q1 = np.dot(dados, H)
    Q2 = np.dot(H, dados.T)
    Q = np.dot(np.dot(Q1, L), Q2)

    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(Q)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados)
    
    return novos_dados


# Implementação não eficiente (oficial)
def KGraphs(dados, labels, k, d, t, lap='padrao'):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     # Esse é o oficial!
            #maiores_autovetores = w[:, ordem[-1:]]      # Pega apenas os 2 primeiros (teste)
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                normais = np.zeros(Wpca.shape)
                curvaturas = np.zeros(m)
                delta = np.zeros(m)
                for k in range(m):
                    normais[:, k] = matriz_pcs[j, :, k] - matriz_pcs[i, :, k]
                    delta[k] = norm(normais[:, k])
                    theta = arccos(dot(matriz_pcs[i, :, k], normais[:, k])/(norm(matriz_pcs[i, :, k])*norm(normais[:, k])))
                    curvaturas[k] = (norm(matriz_pcs[i, :, k])*norm(normais[:, k])*sin(theta))/norm(matriz_pcs[i, :, k])**3
                    if np.isnan(curvaturas[k]):
                        curvaturas[k] = 0.0000001
                if labels[i] == labels[j]:
                    #B[i, j] = min( norm(delta), norm(curvaturas) )
                    B[i, j] = min( min(delta), min(curvaturas) )
                    #B[i, j] = min(delta) + min(curvaturas)
                    #B[i, j] = min(curvaturas)
                else:
                    #B[i, j] = norm(delta) + norm(curvaturas)
                    B[i, j] = max(delta) + max(curvaturas)
                    #B[i, j] = max(curvaturas)
                
    # Aplica kernel Gaussiano 
    W = np.exp(-(B**2)/t)    

    # Matriz diagonal D e Laplaciana L
    D = np.diag(W.sum(1))   # soma as linhas
    L = D - W               # Essa é a matriz Laplaciana entrópica

    if lap == 'normalizada':
        lambdas, alphas = eigh(np.dot(inv(D), L), eigvals=(1, d))
    else:
        lambdas, alphas = eigh(L, eigvals=(1, d))

    return alphas

'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    #print('KNN accuracy: ', acc)

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc = svm.score(X_test, y_test)
    lista.append(acc)
    #print('SVM accuracy: ', acc)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc = nb.score(X_test, y_test)
    lista.append(acc)
    #print('NB accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    #print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    #print('QDA accuracy: ', acc)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc = mpl.score(X_test, y_test)
    lista.append(acc)
    #print('MPL accuracy: ', acc)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc = gpc.score(X_test, y_test)
    lista.append(acc)
    #print('GPC accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    #print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    #print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]



# Plota gráficos de dispersão para o caso 2D
def PlotaDados(dados, labels, metodo):
    
    nclass = len(np.unique(labels))

    if metodo == 'LDA':
        if nclass == 2:
            return -1

    # Converte labels para inteiros
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     # contém as classes (sem repetição)

    # Mapeia rotulos para números
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)

    # Converte para vetor
    rotulos = np.array(rotulos)

    # Obtém o número de classes
    #nclass = len(lista)

    plt.figure(2)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        if (i == 0):
           cor = 'blue'
        elif (i == 1):
            cor = 'red'
        elif (i == 2):
            cor = 'green'
        elif (i == 3):
            cor = 'black'
        elif (i == 4):
            cor = 'orange'
        elif (i == 5):
            cor = 'magenta'
        elif (i == 6):
            cor = 'cyan'
        elif (i == 7):
            cor = 'darkkhaki'
        elif (i == 8):
            cor = 'brown'
        elif (i == 9):
            cor = 'purple'
        else:
            cor = 'salmon'
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='.')
    
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()


#%%%%%%%%%%%%%%%%%%%%  Data loading

#X = skdata.load_iris()     
#X = skdata.load_wine()
#X = skdata.load_breast_cancer()
#X = skdata.fetch_openml(name='tecator', version=2)      
#X = skdata.fetch_openml(name='fl2000', version=1) 
#X = skdata.fetch_openml(name='flags', version=1) 
#X = skdata.fetch_openml(name='vehicle', version=1)  
#X = skdata.fetch_openml(name='mfeat-fourier', version=1) 
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='prnn_fglass', version=1) 
#X = skdata.fetch_openml(name='micro-mass', version=1) 
#X = skdata.fetch_openml(name='diggle_table_a2', version=1) 
#X = skdata.fetch_openml(name='tae', version=1) 
#X = skdata.fetch_openml(name='Smartphone-Based_Recognition_of_Human_Activities', version=1)
#X = skdata.fetch_openml(name='hayes-roth', version=1) 
#X = skdata.fetch_openml(name='glass', version=1) 
#X = skdata.fetch_openml(name='optdigits', version=2) 
#X = skdata.fetch_openml(name='texture', version=1) 
#X = skdata.fetch_openml(name='breast-tissue', version=1) 
#X = skdata.fetch_openml(name='seismic-bumps', version=1) 
#X = skdata.fetch_openml(name='user-knowledge', version=1) 
#X = skdata.fetch_openml(name='vertebra-column', version=1) 
#X = skdata.fetch_openml(name='Engine1', version=1) 
#X = skdata.fetch_openml(name='PopularKids', version=1) 
#X = skdata.fetch_openml(name='heart-h', version=3) 
#X = skdata.fetch_openml(name='anneal', version=4) 
#X = skdata.fetch_openml(name='ecoli', version=3) 
#X = skdata.fetch_openml(name='thyroid-new', version=1) 
#X = skdata.fetch_openml(name='mammography', version=1) 
#X = skdata.fetch_openml(name='spambase', version=1)  
#X = skdata.fetch_openml(name='bank-marketing', version=2) 
#X = skdata.fetch_openml(name='usp05', version=1) 
#X = skdata.fetch_openml(name='autoUniv-au6-750', version=1) 
#X = skdata.fetch_openml(name='cars1', version=1) 
#X = skdata.fetch_openml(name='calendarDOW', version=1) 
#X = skdata.fetch_openml(name='solar-flare', version=3)    
#X = skdata.fetch_openml(name='Touch2', version=1) 
#X = skdata.fetch_openml(name='heart-long-beach', version=1) 
#X = skdata.fetch_openml(name='heart-switzerland', version=1) 
#X = skdata.fetch_openml(name='teachingAssistant', version=1)
#X = skdata.fetch_openml(name='balance-scale', version=1)
#X = skdata.fetch_openml(name='CPMP-2015-runtime-classification', version=1) 
#X = skdata.fetch_openml(name='lymph', version=1) 
#X = skdata.fetch_openml(name='servo', version=1) 
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)
#X = skdata.fetch_openml(name='mux6', version=1) 
#X = skdata.fetch_openml(name='car-evaluation', version=1)          
#X = skdata.fetch_openml(name='wine-quality-white', version=1)
#X = skdata.fetch_openml(name='thyroid-allbp', version=1)
#X = skdata.fetch_openml(name='waveform-5000', version=1)
#X = skdata.fetch_openml(name='segment', version=1)  
#X = skdata.fetch_openml(name='wine-quality-red', version=1)
#X = skdata.fetch_openml(name='cmc', version=1)
#X = skdata.fetch_openml(name='yeast', version=1)
#X = skdata.fetch_openml(name='analcatdata_dmft', version=1)
#X = skdata.fetch_openml(name='rmftsa_sleepdata', version=1)
#X = skdata.fetch_openml(name='steel-plates-fault', version=3)
#X = skdata.fetch_openml(name='vowel', version=2)
X = skdata.fetch_openml(name='diabetes', version=1) 

dados = X['data']
target = X['target']

#%%%%%%%%%%%%%%%%%%%% Supervised classification for ISOMAP-KL features

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
input()

# Only for OpenML datasets (convert categorical features)
# Precisa tratar dados categóricos manualmente
cat_cols = dados.select_dtypes(['category']).columns
dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
# Converte para numpy (openml agora é dataframe)
dados = dados.to_numpy()
target = target.to_numpy()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

#%%%%%%%%%%% Simple PCA 
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=20, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%%%% t-SNE
model = TSNE(n_components=2, perplexity=30)
dados_tsne = model.fit_transform(dados)
dados_tsne = dados_tsne.T

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T

#%%%%%%%%%%%%% Supervised PCA
dados_suppca = SupervisedPCA(dados, target, 2)

#%%%%%%%%%%%% LDA
if c > 2:
    model = LinearDiscriminantAnalysis(n_components=2)
else:
    model = LinearDiscriminantAnalysis(n_components=1)

dados_lda = model.fit_transform(dados, target)
dados_lda = dados_lda.T

#%%%%%%%%%%% Clustering measures
L_pca = Classification(dados_pca, target, 'PCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_tsne = Classification(dados_tsne, target, 't-SNE')
L_umap = Classification(dados_umap, target, 'UMAP')
L_spca = Classification(dados_suppca, target, 'SupervisedPCA')
L_lda = Classification(dados_lda, target, 'LDA')

#################### Geodesic ISOMAP features

# Parameters
k = int(0.75*n)     
t = 1     

print('t = %f' %t)

# Geodesic ISOMAP
dados_klap = KGraphs(dados, target, k, 2, t=t, lap='normalizada')             # No iris, t = 0.0001 foi bom! Com t = 0.00001 foi melhor! No wine, t = 0.001
L_klap = Classification(dados_klap.T, target, 'K-LAPLACIAN')

################################################
############      Plot clusters    #############
################################################
PlotaDados(dados_klap, target, 'K-LAPLACIAN')
PlotaDados(dados_pca.T, target, 'PCA')
PlotaDados(dados_isomap.T, target, 'ISOMAP')
PlotaDados(dados_tsne.T, target, 't-SNE')
PlotaDados(dados_umap.T, target, 'UMAP')
PlotaDados(dados_suppca.T, target, 'SupervisedPCA')
PlotaDados(dados_lda.T, target, 'LDA')