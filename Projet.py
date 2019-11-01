# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:10:35 2019

@author: leo_p
"""

import autograd
from autograd import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg

def find_seed(g,c=0,eps=2**(-26)):
    '''Propose un t qui convient si c'est possible'''
    a=0
    b=1
    if (g(a)-c)*(g(b)-c)>0:    #Si la condition précédente n'est pas vérifée
        return None
    while (b-a)>eps:    #Méthode dichotomique
        t=(b+a)/2
        if (g(a)-c)*(g(t)-c)<=0:
            b=t
        else:
            a=t
    return t

def gradient(f,x,y):
    '''Calcule le gradient de f en (x,y)'''
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]

def simple_contour(f,c=0,delta=0.01):
    '''Donne les coordonnées demandées sur un cadre de côté unitaire de manière optimisée'''
    x=np.array([])
    y=np.array([])
    def g(x):
        return (f(0,x))
    if find_seed(g,c)==None:
        return x,y
    y = np.append(y, np.array([find_seed(g,c)]))

    x = np.append(x, np.array([0]))

    grad=gradient(f,x[-1],y[-1])
    tang=[-grad[1],grad[0]]
    print(tang)
    if tang[0]<0:
        tang=[grad[1],-grad[0]]
        while x[-1] <= 1 and y[-1] <= 1 and x[-1] >= 0 and y[-1] >= 0:
            grad=gradient(f,x[-1],y[-1])
            tang=[grad[1],-grad[0]]
            norme=math.sqrt(tang[0]**2+tang[1]**2)
            x = np.append(x, np.array(x[-1]+(delta*tang[0])/(norme)))
            y = np.append(y, np.array(y[-1]+(delta*tang[1])/(norme)))
            def F(x, y): #On définit la fonction F pour les deux derniers points trouvés
                return np.array([f(x,y) - c, (x-x[-2])**2 + (y-y[-2])**2 - delta**2])
            res = newton(F, np.array([x[-1], y[-1]])) #On applique Newton: ATTENTION, un problème non résolu apparait lors de l'éxécution de autograd.jacobian
            x[-1], y[-1] = res[0], res[1]
    else:
        tang=[-grad[1],grad[0]]
        while x[-1] <= (1) and y[-1] <= (1) and x[-1] >= 0 and y[-1] >= 0:
            grad=gradient(f,x[-1],y[-1])
            tang=[-grad[1],grad[0]]
            norme=math.sqrt(tang[0]**2+tang[1]**2)
            x = np.append(x, np.array(x[-1]+(delta*tang[0])/(norme)))
            y = np.append(y, np.array(y[-1]+(delta*tang[1])/(norme)))
            def F(x, y):
                return np.array([f(x,y) - c, (x-x[-2])**2 + (y-y[-2])**2 - delta**2])
            res = newton(F, np.array([x[-1], y[-1]]))
            x[-1], y[-1] = res[0], res[1]
    return x,y

def f(x,y):
    '''Fonction dont on veut tracer le graphe'''
    return 2*( np.exp(-x**2-y**2)- np.exp(-(x-1)**2 - (y-1)**2))

def J(F, x, y):
    '''Jacobienne de F'''
    j = autograd.jacobian
    return np.c_[j(F, 0)(x, y), j(F, 1)(x, y)]

def newton (F,pt):
    '''Fonction qui éxecute la formule de Newton précédente'''
    jacob = J(F, pt[0], pt[1])
    try:
        Inv = np.linalg.inv(jacob)  #On teste bien si la jacobienne est inversible
    except:
        return(pt) #Sinon, on ne modifie pas le point
    pt_new = pt - np.dot(Inv, F(pt[0], pt[1])) #On applique la formule
    while np.linalg.norm(pt_new - pt) > 2**(-26): #On itère jusqu'à avoir des points qui convergent vers le point fixe recherché
        pt = pt_new
        jacob = J(F, pt[0], pt[1])
        try:
            Inv = np.linalg.inv(jacob)
        except:
            return (pt)
        pt_new = pt - np.dot(Inv, F(pt[0], pt[1]))
    return (pt_new)

# Rotators
# ------------------------------------------------------------------------------
LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3  # clockwise


def rotate_direction(direction, n=1):
    return (direction + n) % 4


def rotate(x, y, n=1):
    if n == 0:
        return x, y
    elif n >= 1:
        return rotate(1 - y, x, n - 1)
    else:
        assert n < 0
        return rotate(x, y, n=-3 * n)


def rotate_function(f, n=1):
    def rotated_function(x, y):
        xr, yr = rotate(x, y, -n)
        return f(xr, yr)

    return rotated_function

# Complex Contouring
# ------------------------------------------------------------------------------

# Customize the simple_contour function used in contour :
# simple_contour = smart_simple_contour


def contour(f, c, xs=[0.0, 1.0], ys=[0.0, 1.0], delta=0.01):
    curves = []
    nx, ny = len(xs), len(ys)
    for i in range(nx - 1):
        for j in range(ny - 1):
            xmin, xmax = xs[i], xs[i + 1]
            ymin, ymax = ys[j], ys[j + 1]

            def f_cell(x, y):
                return f(xmin + (xmax - xmin) * x, ymin + (ymax - ymin) * y)

            done = set()
            for n in [0, 1, 2, 3]:
                if n not in done:
                    rotated_f_cell = rotate_function(f_cell, n)
                    x_curve_r, y_curve_r = simple_contour(rotated_f_cell, c, delta) #Nous n'avons donc pas pu utiliser la fonction newton pour optimiser le tracé de fonction
                    exit = None
                    if len(x_curve_r) >= 1:
                        xf, yf = x_curve_r[-1], y_curve_r[-1]
                        if xf == 0.0:
                            exit = LEFT
                        elif xf == 1.0:
                            exit = RIGHT
                        elif yf == 0.0:
                            exit = DOWN
                        elif yf == 1.0:
                            exit = UP
                    if exit is not None:  # a fully successful contour fragment
                        exit = rotate_direction(exit, n)
                        done.add(exit)

                    x_curve, y_curve = [], []
                    for x_r, y_r in zip(x_curve_r, y_curve_r):
                        x, y = rotate(x_r, y_r, n=-n)
                        x_curve.append(x)
                        y_curve.append(y)
                    x_curve = np.array(x_curve)
                    y_curve = np.array(y_curve)
                    curves.append(
                        (xmin + (xmax - xmin) * x_curve, ymin + (ymax - ymin) * y_curve)
                    )
    return curves
