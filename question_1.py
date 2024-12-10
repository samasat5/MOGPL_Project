
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt
from gurobipy import *
import networkx as nx
import gurobipy as gp



def ex1_maximin_approach():

    maximin = Model("1_MaxiMin")

    x1 = maximin.addVar(vtype=GRB.BINARY, name="x1")
    x2 = maximin.addVar(vtype=GRB.BINARY, name="x2")
    x3 = maximin.addVar(vtype=GRB.BINARY, name="x3")
    x4 = maximin.addVar(vtype=GRB.BINARY, name="x4")
    x5 = maximin.addVar(vtype=GRB.BINARY, name="x5")
    x6 = maximin.addVar(vtype=GRB.BINARY, name="x6")
    x7 = maximin.addVar(vtype=GRB.BINARY, name="x7")
    x8 = maximin.addVar(vtype=GRB.BINARY, name="x8")
    x9 = maximin.addVar(vtype=GRB.BINARY, name="x9")
    x10= maximin.addVar(vtype=GRB.BINARY, name="x10")
    g  = maximin.addVar(name="g")


    # objective function
    maximin.setObjective(g, GRB.MAXIMIZE)


    # Add constraints
    maximin.addConstr(g <= 70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10, "c0")  # miniming over the scenarios
    maximin.addConstr(g <= 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10, "c1")  # miniming over the scenarios
    maximin.addConstr(60 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5 + 20 * x6 + 5 * x7 + 15 * x8 + 20 * x9 + 60 * x10 <= 100, "c2")  # budget constraint


    # Optimize the model
    maximin.optimize()

    # Print the results
    if maximin.status == GRB.OPTIMAL:
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = int(x1.X), int(x2.X), int(x3.X), int(x4.X), int(x5.X), int(x6.X), int(x7.X), int(x8.X), int(x9.X), int(x10.X)
        x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
        z_1 = 70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10
        z_2 = 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10
        z = (z_1, z_2)
        print("\n\n\n1.1: Optimal solution found for Maximin approach:")
        print(f"x* = {x} ")
        print(f"g* = {int(g.X)} ")
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = 1,0,1,0,0,0,1,1,0,0
        print(f"Objective value: {int(maximin.objVal)}")
        print(f"(z_1, z_2) = {(z_1, z_2)} ")
        print(f"Checking non-optimal examples: \nif x = [1,0,1,0,0,0,1,1,0,0] :")
        z_1 = 70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10
        z_2 = 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10
        z = (z_1, z_2)
        print (f"then z = (z_1, z_2) = {z}, which g(x) = {min(z_1, z_2)} is lower than found g(x)* = {int(g.X)}, therefore its not optimal\n\n\n")

    else:
        print("No optimal solution found.")
    
    return x






## Getting the z_1* :
def ex1_minmax_regret_approach_optz1 ():

    minmax_regret_s1 = Model("1_Minimax_Regret, for getting z*1")

    x1_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x1")
    x2_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x2")
    x3_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x3")
    x4_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x4")
    x5_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x5")
    x6_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x6")
    x7_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x7")
    x8_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x8")
    x9_s1 = minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x9")
    x10_s1= minmax_regret_s1.addVar(vtype=GRB.BINARY, name="x10")

    minmax_regret_s1.setObjective(70 * x1_s1 + 18 * x2_s1 + 16 * x3_s1 + 14 * x4_s1 + 12 * x5_s1 + 10 * x6_s1 + 8 * x7_s1 + 6 * x8_s1 + 4 * x9_s1 + 2 * x10_s1, GRB.MAXIMIZE)

    minmax_regret_s1.addConstr(60 * x1_s1 + 10 * x2_s1 + 15 * x3_s1 + 20 * x4_s1 + 25 * x5_s1 + 20 * x6_s1 + 5 * x7_s1 + 15 * x8_s1 + 20 * x9_s1 + 60 * x10_s1 <= 100, "c2")  # budget constraint
    minmax_regret_s1.optimize()

    if minmax_regret_s1.status == GRB.OPTIMAL:
        x1_s1, x2_s1, x3_s1, x4_s1, x5_s1, x6_s1, x7_s1, x8_s1, x9_s1, x10_s1 = int(x1_s1.X), int(x2_s1.X), int(x3_s1.X), int(x4_s1.X), int(x5_s1.X), int(x6_s1.X), int(x7_s1.X), int(x8_s1.X), int(x9_s1.X), int(x10_s1.X)
        x_s1 = [x1_s1, x2_s1, x3_s1, x4_s1, x5_s1, x6_s1, x7_s1, x8_s1, x9_s1, x10_s1]
        print("\n\n\n1.2: Optimal solution found for z_1 *:")
        print(f"x_s1* = {x_s1} ")
        opt_z_1 = int(minmax_regret_s1.objVal)
        print(f"Objective value: {opt_z_1 =} \n\n\n")

    else:
        print("No optimal solution found.")

    return x_s1, opt_z_1



## Getting the z_2* :
def ex1_minmax_regret_approach_optz2 ():
    minmax_regret_s2 = Model("1_Minimax_Regret, for getting z*2")

    x1_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x1")
    x2_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x2")
    x3_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x3")
    x4_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x4")
    x5_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x5")
    x6_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x6")
    x7_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x7")
    x8_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x8")
    x9_s2 = minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x9")
    x10_s2= minmax_regret_s2.addVar(vtype=GRB.BINARY, name="x10")

    minmax_regret_s2.setObjective(2 * x1_s2 + 4 * x2_s2 + 6 * x3_s2 + 8 * x4_s2 + 10 * x5_s2 + 12 * x6_s2 + 14 * x7_s2 + 16 * x8_s2 + 18 * x9_s2 + 70 * x10_s2, GRB.MAXIMIZE)
    minmax_regret_s2.addConstr(60 * x1_s2 + 10 * x2_s2 + 15 * x3_s2 + 20 * x4_s2 + 25 * x5_s2 + 20 * x6_s2 + 5 * x7_s2 + 15 * x8_s2 + 20 * x9_s2 + 60 * x10_s2 <= 100, "c2")  # budget constraint
    minmax_regret_s2.optimize()

    if minmax_regret_s2.status == GRB.OPTIMAL:
        x1_s2, x2_s2, x3_s2, x4_s2, x5_s2, x6_s2, x7_s2, x8_s2, x9_s2, x10_s2 = int(x1_s2.X), int(x2_s2.X), int(x3_s2.X), int(x4_s2.X), int(x5_s2.X), int(x6_s2.X), int(x7_s2.X), int(x8_s2.X), int(x9_s2.X), int(x10_s2.X)
        x_s2 = [x1_s2, x2_s2, x3_s2, x4_s2, x5_s2, x6_s2, x7_s2, x8_s2, x9_s2, x10_s2]
        print("\n\n\n1.2: Optimal solution found for z_2 *:")
        print(f"x_s2* = {x_s2} ")
        opt_z_2 = int(minmax_regret_s2.objVal)
        print(f"Objective value: {opt_z_2 =}\n\n\n")

    else:
        print("No optimal solution found.")
    return x_s2, opt_z_2



def ex1_minmax_regret_approach(opt_z_1, opt_z_2):

    minmax_regret = Model("1_Minimax_Regret")


    x1 = minmax_regret.addVar(vtype=GRB.BINARY, name="x1")
    x2 = minmax_regret.addVar(vtype=GRB.BINARY, name="x2")
    x3 = minmax_regret.addVar(vtype=GRB.BINARY, name="x3")
    x4 = minmax_regret.addVar(vtype=GRB.BINARY, name="x4")
    x5 = minmax_regret.addVar(vtype=GRB.BINARY, name="x5")
    x6 = minmax_regret.addVar(vtype=GRB.BINARY, name="x6")
    x7 = minmax_regret.addVar(vtype=GRB.BINARY, name="x7")
    x8 = minmax_regret.addVar(vtype=GRB.BINARY, name="x8")
    x9 = minmax_regret.addVar(vtype=GRB.BINARY, name="x9")
    x10= minmax_regret.addVar(vtype=GRB.BINARY, name="x10")
    r  = minmax_regret.addVar(name="r")





    # Add constraints
    minmax_regret.addConstr(r >= opt_z_1 - (70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10), "c0")  # miniming over the scenarios
    minmax_regret.addConstr(r >= opt_z_2 - (2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10), "c1")  # miniming over the scenarios
    minmax_regret.addConstr(60 * x1 + 10 * x2 + 15 * x3 + 20 * x4 + 25 * x5 + 20 * x6 + 5 * x7 + 15 * x8 + 20 * x9 + 60 * x10 <= 100, "c2")  # budget constraint


    minmax_regret.setObjective( r, GRB.MINIMIZE)



    # Optimize the model
    minmax_regret.optimize()

    # Print the results
    if minmax_regret.status == GRB.OPTIMAL:
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = int(x1.X), int(x2.X), int(x3.X), int(x4.X), int(x5.X), int(x6.X), int(x7.X), int(x8.X), int(x9.X), int(x10.X)
        x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
        print("\n\n\n1.2: Optimal solution found for Minimax regret approach:")
        print(f"x* = {x} ")
        print(f"Objective value: {int(minmax_regret.objVal)}")
        z_1 = 70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10
        z_2 = 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10
        print(f"(z_1, z_2) = {(z_1, z_2)} ")
        r_1 = opt_z_1 - (70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10)
        r_2 = opt_z_1 - (2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10)
        print (f"(r_1, r_2) = ({r_1}, {r_2}) therefore g(x)* = {max(r_2,r_1)} \n\n")
    

        # print(f"Checking non-optimal examples: \nif x = [1,0,1,0,0,0,1,1,0,0] :")
        # z_1 = 70 * x1 + 18 * x2 + 16 * x3 + 14 * x4 + 12 * x5 + 10 * x6 + 8 * x7 + 6 * x8 + 4 * x9 + 2 * x10
        # z_2 = 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4 + 10 * x5 + 12 * x6 + 14 * x7 + 16 * x8 + 18 * x9 + 70 * x10
        # z = (z_1, z_2)
        # print (f"then z = (z_1, z_2) = {z}, which g = {min(z_1, z_2)} is lower than found g* = {int(g.X)}, therefore its not optimal")

    else:
        print("No optimal solution found.")
    return x












def ex1_maximin_and_minimaxregret_check (x_star, xprime_star, x_s1, x_s2):

    x_star_x1, x_star_x2, x_star_x3, x_star_x4, x_star_x5, x_star_x6, x_star_x7, x_star_x8, x_star_x9, x_star_x10 = x_star
    xprime_star_x1, xprime_star_x2, xprime_star_x3, xprime_star_x4, xprime_star_x5, xprime_star_x6, xprime_star_x7, xprime_star_x8, xprime_star_x9, xprime_star_x10,  = xprime_star
    x1_s1, x2_s1, x3_s1, x4_s1, x5_s1, x6_s1, x7_s1, x8_s1, x9_s1, x10_s1 = x_s1
    x1_s2, x2_s2, x3_s2, x4_s2, x5_s2, x6_s2, x7_s2, x8_s2, x9_s2, x10_s2 = x_s2
    
    # For z_1_x_star and z_2_x_star
    z_1_x_star = 70 * x_star_x1 + 18 * x_star_x2 + 16 * x_star_x3 + 14 * x_star_x4 + 12 * x_star_x5 + 10 * x_star_x6 + 8 * x_star_x7 + 6 * x_star_x8 + 4 * x_star_x9 + 2 * x_star_x10
    z_2_x_star = 2 * x_star_x1 + 4 * x_star_x2 + 6 * x_star_x3 + 8 * x_star_x4 + 10 * x_star_x5 + 12 * x_star_x6 + 14 * x_star_x7 + 16 * x_star_x8 + 18 * x_star_x9 + 70 * x_star_x10

    # For z_1_xprime and z_2_xprime
    z_1_xprime = 70 * xprime_star_x1 + 18 * xprime_star_x2 + 16 * xprime_star_x3 + 14 * xprime_star_x4 + 12 * xprime_star_x5 + 10 * xprime_star_x6 + 8 * xprime_star_x7 + 6 * xprime_star_x8 + 4 * xprime_star_x9 + 2 * xprime_star_x10
    z_2_xprime = 2 * xprime_star_x1 + 4 * xprime_star_x2 + 6 * xprime_star_x3 + 8 * xprime_star_x4 + 10 * xprime_star_x5 + 12 * xprime_star_x6 + 14 * xprime_star_x7 + 16 * xprime_star_x8 + 18 * xprime_star_x9 + 70 * xprime_star_x10

    # For z_1_x_s1 and z_2_x_s1
    z_1_x_s1 = 70 * x1_s1 + 18 * x2_s1 + 16 * x3_s1 + 14 * x4_s1 + 12 * x5_s1 + 10 * x6_s1 + 8 * x7_s1 + 6 * x8_s1 + 4 * x9_s1 + 2 * x10_s1
    z_2_x_s1 = 2 * x1_s1 + 4 * x2_s1 + 6 * x3_s1 + 8 * x4_s1 + 10 * x5_s1 + 12 * x6_s1 + 14 * x7_s1 + 16 * x8_s1 + 18 * x9_s1 + 70 * x10_s1

    # For z_1_x_s2 and z_2_x_s2
    z_1_x_s2 = 70 * x1_s2 + 18 * x2_s2 + 16 * x3_s2 + 14 * x4_s2 + 12 * x5_s2 + 10 * x6_s2 + 8 * x7_s2 + 6 * x8_s2 + 4 * x9_s2 + 2 * x10_s2
    z_2_x_s2 = 2 * x1_s2 + 4 * x2_s2 + 6 * x3_s2 + 8 * x4_s2 + 10 * x5_s2 + 12 * x6_s2 + 14 * x7_s2 + 16 * x8_s2 + 18 * x9_s2 + 70 * x10_s2


    
    point_x_star = (z_1_x_star, z_2_x_star)
    point_xprime_star = (z_1_xprime, z_2_xprime)
    point_z1_star = (z_1_x_s1, z_2_x_s1)
    point_z2_star = (z_1_x_s2, z_2_x_s2)


    points = {"x*": point_x_star, "x'*": point_xprime_star, "x1*": point_z1_star, "x2*": point_z2_star}

    print("1.3: Checking and ploting the x*, x'*, x1* and x2* points in the z_1-z_2 Plane:")
    for label, (z1, z2) in points.items():
        print(f"{label}: z_1 = {z1}, z_2 = {z2}")


    z1_x1, z2_x1 = points["x1*"]
    z1_x2, z2_x2 = points["x2*"]

        
    # Coordinates of x1* and x2* (replace with actual values)
    x1_star = (z_1_x_s1, z_2_x_s1)
    x2_star = (z_1_x_s2, z_2_x_s2)

    # Coordinates of x* and x'*
    x_star = (z_1_x_star, z_2_x_star)
    x_prime_star = (z_1_xprime, z_2_xprime)


    for label, (z1, z2) in points.items():
        plt.scatter(z1, z2, label=label)
        plt.text(z1, z2, f"({z1}, {z2})", fontsize=9, ha='right')

    plt.plot([z1_x1, z1_x2], [z2_x1, z2_x2], linestyle='--', color='gray', label="w1*z1_star + w2*z2_star")


    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.title("Points in the z_1-z_2 Plane")
    plt.legend()
    plt.grid(True)

    plt.show()


def resolution_time (n, p, num_instances):
    
    resolution_times = []

    for instance in range(num_instances):

        coeff_z_i = np.random.randint(1, 100, size=(n, p))
        cost = np.random.randint(1, 100, size=(1, p))
        budget = (np.sum(cost)) * 0.5


        start_time = time.time()


        model = Model(f"Instance_{instance + 1}")

        x = model.addVars(p, vtype=GRB.BINARY, name="x")
        g = model.addVar(name="g")

        model.setObjective(g, GRB.MAXIMIZE)


        for scenario in range(n):
            model.addConstr(g <= sum(coeff_z_i[scenario, j] * x[j] for j in range(p)), name=f"c{scenario}")
        model.addConstr(sum(cost[0, j] * x[j] for j in range(p)) <= budget, name="budget")

        model.optimize()


        end_time = time.time()


        resolution_times.append(end_time - start_time)


        if model.status == GRB.OPTIMAL:
            print(f"\n\nInstance {instance + 1}: Solution found.")
            print(f"  Objective value: {int(model.objVal)}")
            x_values = [int(x[j].X) for j in range(p)]
            print(f"  Selected projects: {x_values}\n\n")
        else:
            print(f"Instance {instance + 1}: No optimal solution found.\n\n")


    average_time = sum(resolution_times) / len(resolution_times)
    print(f"\nAverage Resolution Time for {num_instances} instances in a model with {p} projects and {n} scenarios: {average_time:.4f} seconds\n\n")
    return average_time








def plot_heatmap(results, scenarios, projects):

    heatmap_data = np.zeros((len(scenarios), len(projects)))
    for i, n in enumerate(scenarios):
        for j, p in enumerate(projects):
            heatmap_data[i, j] = results[(n, p)]


    fig, ax = plt.subplots()
    cax = ax.imshow(heatmap_data, cmap='grey', origin='upper', aspect='auto')

    cbar = fig.colorbar(cax)
    cbar.set_label('Average Resolution Time (s)', rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(projects)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(projects)
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("p, number of projects/nodes")
    ax.set_ylabel("n, number of scenarios")
    ax.set_title("Average Resolution Time heatmap for different (n,p)")
    plt.show()



def comparing_different_scenario_project_combinations():
    scenarios = [15, 10, 5]
    projects = [10, 15, 20]
    num_instances = 10  

    results = {}

    for n in scenarios:
        for p in projects:
            avg_time = resolution_time(n, p, num_instances)
            results[(n, p)] = avg_time


    print("\nSummary of Average Resolution Times:\n")
    for (n, p), avg_time in results.items():
        print(f"\nScenarios: {n}, Projects: {p} -> Avg Time: {avg_time:.4f} seconds\n\n")

    print(f'\nresults = {results}\n') 
    plot_heatmap(results, scenarios, projects) 










def run_question_1():
    
    x_star = ex1_maximin_approach()  

    x_s1, opt_z_1 = ex1_minmax_regret_approach_optz1()  
    x_s2, opt_z_2 = ex1_minmax_regret_approach_optz2()  
    xprime_star = ex1_minmax_regret_approach(opt_z_1, opt_z_2)  
    
    ex1_maximin_and_minimaxregret_check (x_star, xprime_star, x_s1, x_s2)  
    
    comparing_different_scenario_project_combinations() 
