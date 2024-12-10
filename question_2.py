from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gurobipy import *
import networkx as nx
import gurobipy as gp
import question_1
from question_1 import run_question_1


def exo2_2(k):
    
    
    nbcont=6
    nbvar=7
    
    # Range of plants and warehouses
    lignes = range(nbcont)
    colonnes = range(nbvar)
    
    
    # Matrice des contraintes
    a = [[1, -1, 0, 0, 0, 0, 0],
         [1, 0, -1, 0, 0, 0, 0],
         [1, 0, 0, -1, 0, 0, 0],
         [1, 0, 0, 0, -1, 0, 0],
         [1, 0, 0, 0, 0, -1, 0],
         [1, 0, 0, 0, 0, 0, -1]]
    
    # Second membre
    b = [2, 9, 6, 8, 5, 4]
    
    # Coefficients de la fonction objectif
    c = [k, -1, -1, -1, -1, -1, -1]
    
    m = Model("question_2_2")
    
    # declaration variables de decision
    x = []
    x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="r%d" % k))
    for i in range(1,nbvar):
        x.append(m.addVar(vtype=GRB.INTEGER, lb=0, name="b%d%d" % ((i), k)))
    
    # maj du modele pour integrer les nouvelles variables
    m.update()
    
    obj = LinExpr();
    obj =0
    for j in colonnes:
        obj += c[j] * x[j]
    
    # definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)
    
    # Definition des contraintes
    for i in lignes:
        m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)
    
    # Resolution
    m.optimize()
    
    
    print("")
    print('Optimal solution:')
    
    print('r%d'%(k), '=', x[0].x)
    for j in range(1,nbvar):
        print('b%d%d'%((j),k), '=', x[j].x)
    print("")
    print('Objective value :', m.objVal)
    return m.objVal



def exo2_4():
    nbcont=5
    nbvar=16
    
    # Range of plants and warehouses
    lignes = range(nbcont)
    colonnes = range(nbvar)
    
    # Matrice des contraintes
    a = [[0, 0, 0, 0, 0, 0, 60, 10, 15, 20, 25, 20, 5, 15, 20, 60],
         [1, 0, -1, 0, 0, 0, -70, -18, -16, -14, -12, -10, -8, -6, -4, -2],
         [1, 0, 0, 0, -1, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -70],
         [0, 1, 0, -1, 0, 0, -70, -18, -16, -14, -12, -10, -8, -6, -4, -2],
         [0, 1, 0, 0, 0, -1, -2, -4, -6, -8, -10, -12, -14, -16, -18, -70]]
    
    # Second membre
    b = [100, 0, 0, 0, 0]
    
    # Coefficients de la fonction objectif
    c = [1, 2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    m = Model("question_2_4")
    
    # declaration variables de decision
    x = []
    x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,name="r1"))
    x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="r2"))
    for i in range(2):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d1" % (i+1)))
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d2" % (i+1)))
    for i in range(10):
        x.append(m.addVar(vtype=GRB.BINARY, lb=0, name="x%d" % (i+1)))
    
    # maj du modele pour integrer les nouvelles variables
    m.update()
    
    obj = LinExpr();
    obj = 0
    for j in colonnes:
        obj += c[j] * x[j]
    
    # definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)
    
    # Definition des contraintes
    #m.addConstr(quicksum(a[0][j]*x[j] for j in colonnes) == b[0], "Contrainte0")
    for i in range(0,5):
        m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)
    
    # Resolution
    m.optimize()
    
    print("")
    print('Solution optimale:')
    print('r1', '=', x[0].x)
    print('r2', '=', x[1].x)
    print('b11', '=', x[2].x)
    print('b12', '=', x[3].x)
    print('b21', '=', x[4].x)
    print('b22', '=', x[5].x)
    print("\nSelection des objets : ")
    for j in range(6,16):
        print('x%d'%(j-5), '=', x[j].x)
    print("")
    print('Valeur de la fonction objectif :', m.objVal)
    print("z1 =", 5*x[6].x + 6*x[7].x + 4*x[8].x + 8*x[9].x + x[10].x )
    print("z2 =", 3*x[6].x + 8*x[7].x + 6*x[8].x + 2*x[9].x + 5*x[10].x )






def solve_instance(n, p, criterion):
    model = Model("Robust_Knapsack")
    
    # Génération aléatoire des données
    costs = np.random.randint(1, 101, size=n)
    utilities = np.random.randint(1, 101, size=(n, p))
    budget = int(0.5 * np.sum(costs))
   
    # Variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    z = model.addVars(p, vtype=GRB.CONTINUOUS, name="z")
    r = model.addVars(p, vtype=GRB.CONTINUOUS, name="r")
    
    # Contrainte de budget
    model.addConstr(quicksum(costs[i] * x[i] for i in range(n)) <= budget, "Budget")
    
    # Critères
    if criterion == "maxOWA":
        for i in range(p):
            model.addConstr(z[i] == quicksum(utilities[j][i] * x[j] for j in range(n)), f"Utility_{i}")
        model.setObjective(quicksum(z[i] for i in range(p)), GRB.MAXIMIZE)
    
    elif criterion == "minOWA":
        z_star = [utilities[:, i].max() for i in range(p)]
        for i in range(p):
            model.addConstr(r[i] == z_star[i] - quicksum(utilities[j][i] * x[j] for j in range(n)), f"Regret_{i}")
        model.setObjective(quicksum(r[i] for i in range(p)), GRB.MINIMIZE)
    
    # Résolution
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    return solve_time    
    



def minOWA(z1_opt, z2_opt):

    # Données du problème
    utilite_s1 = [70, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    utilite_s2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 70]
    couts = [60, 10, 15, 20, 25, 20, 5, 15, 20, 60]
    budget = 100
    poids = [2, 1]       # Poids OWA
    
    # Initialisation du modèle
    m = Model("minOWA_regrets")
    
    # Variables
    x = m.addVars(10, vtype=GRB.BINARY, name="x")  # Variables de décision
    r = m.addVars(2, vtype=GRB.CONTINUOUS, name="r")  # Regrets triés
    regrets = m.addVars(2, vtype=GRB.CONTINUOUS, name="regrets")  # Regrets non triés
    
    # Contraintes
    # 1. Définition des regrets
    m.addConstr(regrets[0] == z1_opt - quicksum(utilite_s1[j] * x[j] for j in range(10)))
    m.addConstr(regrets[1] == z2_opt - quicksum(utilite_s2[j] * x[j] for j in range(10)))
    
    # 2. Tri des regrets
    for i in range(2):
        for k in range(2):
            m.addConstr(r[k] >= regrets[i])
    
    # 3. Contrainte budgétaire
    m.addConstr(quicksum(couts[j] * x[j] for j in range(10)) <= budget)
    
    # Fonction objectif : Min OWAs des regrets
    m.setObjective(quicksum(poids[k] * r[k] for k in range(2)), GRB.MINIMIZE)
    
    # Résolution
    m.optimize()
    
    # Extraction des résultats
    if m.status == GRB.OPTIMAL:
        print("Solution optimale trouvée")
        print("Projets sélectionnés :")
        for j in range(10):
            if x[j].x > 0.5:
                print(f"Projet {j+1}")
        print("Valeur optimale de la fonction objectif :", m.objVal)
    else:
        print("Pas de solution optimale trouvée")
    



def comparing_minOWA_maxOWA():
    scenarios = [5, 10, 15]
    projects = [10, 15, 20]
    num_instances = 10  

    results = {}

    for n in scenarios:
        for p in projects:
            total_time_minowa = 0
            total_time_maxowa = 0
            for _ in range(num_instances):
                total_time_minowa += solve_instance(n, p, "minOWA")
                total_time_maxowa += solve_instance(n, p, "maxOWA")
            results[(n, p, "minOWA")] = total_time_minowa / num_instances
            results[(n, p, "maxOWA")] = total_time_maxowa / num_instances

    # Affichage des résultats
    print("\nSummary of Average Resolution Times:\n")
    for (n, p, criterion), avg_time in results.items():
        print(f"Scenarios: {n}, Projects: {p}, Criterion: {criterion} -> Avg Time: {avg_time:.4f} seconds")






def run_question_2():
    
    z = np.zeros(6)
    for k in range(1,7):
        z[k-1] = exo2_2(k)
    print("z = ", z)
    exo2_4()
    # _, opt_z_1 = question_1.ex1_minmax_regret_approach_optz1() 
    # _, opt_z_2 = question_1.ex1_minmax_regret_approach_optz2() 
    # minOWA(z1_opt=opt_z_1, z2_opt=opt_z_2)
    # comparing_minOWA_maxOWA()