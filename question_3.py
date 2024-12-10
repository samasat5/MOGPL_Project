
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gurobipy import *
import networkx as nx
import gurobipy as gp
import random
import pdb




# Left Graph information: 
EDGES_LEFT_GRAPH = [(0,1), (0,2), (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (3,5), (4,5)]
D_S1_LEFT_GRAPH = np.array([
    [0, 4, 5, 0, 0, 0],
    [0, 0, 2, 1, 2, 7],
    [0, 0, 0, 5, 2, 0],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0]])
D_S2_LEFT_GRAPH = np.array([
    [0, 3, 1, 0, 0, 0],
    [0, 0, 1, 4, 2, 5],
    [0, 0, 0, 1, 7, 0],
    [0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]])

# Right Graph information:
EDGES_RIGHT_GRAPH = [(0,1), (0,2), (0,3), (1,2), (1,2),(1,3), (1,4), (2,5), (2,4),(3,2), (3,5), (4,6), (5,6)]
D_S1_RIGHT_GRAPH = np.array([
    [0, 5, 10, 2, 0, 0, 0],
    [0, 0, 4, 1, 4, 0, 0],
    [0, 0, 0, 0, 3, 1, 0],
    [0, 0, 1, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1]])
D_S2_RIGHT_GRAPH = np.array([
    [0, 3, 4, 6, 0, 0, 0],
    [0, 0, 2, 3, 6, 0, 0],
    [0, 0, 0, 0, 1, 2, 0],
    [0, 0, 4, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1]])

GRAPH = ["left", "right"]
SCENARIO = [1, 2]



def ex3_1_opt_for_each_senario(edges, D_s, scenario, graph, dest_node):
    
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    m = gp.Model()
    x = m.addVars(G.edges)
    m.update() 
    

    m.setObjective(gp.quicksum(D_s[i,j] * x[i,j] for i,j in G.edges), GRB.MINIMIZE)

        
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0 
        for i in G.nodes if i not in [0,dest_node]),"flow")
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 1 
        for i in G.nodes if i==dest_node),"if k=t")
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) ==-1 
        for i in G.nodes if i==0),"if k=s")
    


    m.optimize()
    
    optimal_solution_matrix = np.zeros((len(D_s), len(D_s[0])))

    if m.status == GRB.OPTIMAL:
        print(f"\n\nQ 3.2: Optimal solution found for scenario {scenario} of the {graph} graph:")

        solution_edges = [(i, j) for i, j in G.edges if x[i, j].X > 0.5]
        print("The path:", solution_edges)
        print(f'\n')
        total_distance = 0
        list_of_t = []
        for i, j in G.edges():
            if x[i, j].X > 0.5:
                optimal_solution_matrix[i, j] = x[i, j].X 
                total_distance += D_s[i,j]
                list_of_t.append(D_s[i,j])
        
        print (f"list of t_ij = {list_of_t}\n")
        print(f'total distance:{total_distance}\n')
        print("x*:")
        print(optimal_solution_matrix )
        print(f'\n')
 
        
        return total_distance




def ex3_2_maximin(edges, D_s1, D_s2, dest_node):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    m = gp.Model("maximin")

    # Variables
    x = m.addVars(G.edges, vtype=GRB.BINARY, name="x")
    g = m.addVar(vtype=GRB.CONTINUOUS, name="g")
    

    # Objective: 
    m.setObjective(g, GRB.MAXIMIZE)
    
    #  constraints 
    m.addConstr(g <= gp.quicksum(D_s1[i, j] * x[i, j] for i, j in G.edges if D_s1[i, j] > 0))
    m.addConstr(g <= gp.quicksum(D_s2[i, j] * x[i, j] for i, j in G.edges if D_s2[i, j] > 0))


    
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0 
        for i in G.nodes if i not in [0,dest_node]),"flow")
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 1 
        for i in G.nodes if i==dest_node),"if k=t")
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) ==-1 
        for i in G.nodes if i==0),"if k=s")
    # m.addConstr(gp.quicksum(x[0, j] for j in G.successors(0)) == 1, "source_flow")
    # m.addConstr(gp.quicksum(x[j, dest_node] for j in G.predecessors(dest_node)) == 1, "sink_flow")
        

    m.optimize()


    path_maximin = []
    optimal_solution_matrix = np.zeros((len(D_s1), len(D_s1[0])))


    
    if m.status == GRB.OPTIMAL:
        print(f'\n\n\nQ 3.3: Optimal solution found for Maximin approach')
        print("g:", g.X)
        print(f"\nObjective value: {m.objVal:.6f}\n")
        print("Edges in the path:")
        for i, j in G.edges():
            if x[i, j].X > 0.5:
                print(f"Edge ({i}, {j}) with capacities D_s1: {D_s1[i, j]} and D_s2: {D_s2[i, j]}")
                optimal_solution_matrix[i, j] = x[i, j].X  
                path_maximin.append((i,j))
                
        print(f'\noptimal Soluton is:\n{optimal_solution_matrix}\n')
        print(f'{path_maximin =}\n')
             

def ex3_2_minimax_regret(edges, D_s1, D_s2, total_t_scenario1, total_t_scenario2, dest_node):

    G = nx.DiGraph()
    G.add_edges_from(edges)

    m = gp.Model("minimax_regret")

    x = m.addVars(G.edges, vtype=GRB.BINARY, name="x")
    r = m.addVar(vtype=GRB.CONTINUOUS, name="r")

    m.setObjective(r, GRB.MINIMIZE)
    

    m.addConstr(r >= total_t_scenario1 - gp.quicksum(D_s1[i, j] * x[i, j] for i, j in G.edges if D_s1[i, j] > 0))
    m.addConstr(r >= total_t_scenario2 - gp.quicksum(D_s2[i, j] * x[i, j] for i, j in G.edges if D_s2[i, j] > 0))
    
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0 
        for i in G.nodes if i not in [0,dest_node]),"flow")
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 1 
        for i in G.nodes if i==dest_node),"if k=t")
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) ==-1 
        for i in G.nodes if i==0),"if k=s")
    
    m.optimize()

    path_minimax_regret = []
    optimal_solution_matrix = np.zeros((len(D_s1), len(D_s1[0])))
    if m.status == GRB.OPTIMAL:
        print(f'\n\n\nQ 3.3: Optimal solution found for Minimax regret approach')
        print("r=", r.X)
        print(f"\nObjective value: {int(m.objVal)}\n")
        print("Edges in the path:")
        for i, j in G.edges():
            if x[i, j].X > 0.5:
                print(f"Edge ({i}, {j}) with capacities   D_s1: {D_s1[i, j]}, D_s2: {D_s2[i, j]}")
                optimal_solution_matrix[i, j] = x[i, j].X  
                path_minimax_regret.append((i,j) )
        
        print(f"\nOptimal Solution is: \n{optimal_solution_matrix}\n")
        print(f'{path_minimax_regret =}\n\n')
        




def ex3_3_maxOWA(edges, D_s1, D_s2, weights):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    m = gp.Model("maxOWA_3_3")

    x = m.addVars(G.edges, vtype=GRB.BINARY, name="x")

    z = m.addVars(2, vtype=GRB.CONTINUOUS, name="z")

    r = m.addVars(2, vtype=GRB.CONTINUOUS, name="r")
    
    dest_node = max(G.nodes)

    m.addConstrs(
        (gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0
         for i in G.nodes if i not in [0, dest_node]), "flow")
    m.addConstr((gp.quicksum(x[j, dest_node] for j in G.predecessors(dest_node)) == 1), "sink")
    m.addConstr((gp.quicksum(x[0, j] for j in G.successors(0)) == 1), "source")

    m.addConstr(z[0] == gp.quicksum(D_s1[i, j] * x[i, j] for i, j in G.edges), "z_s1")
    m.addConstr(z[1] == gp.quicksum(D_s2[i, j] * x[i, j] for i, j in G.edges), "z_s2")

    for i in range(2):
        m.addConstr(r[0] >= z[i], f"sort_r1_geq_z{i}")
        m.addConstr(r[1] <= z[i], f"sort_r2_leq_z{i}")
    m.addConstr(r[0] + r[1] == z[0] + z[1], "sum_r_equals_sum_z")

    m.setObjective(gp.quicksum(weights[i] * r[i] for i in range(2)), GRB.MAXIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("\nOptimal solution for maxOWA (3.3):")
        for i, j in G.edges:
            if x[i, j].X > 0.5:
                print(f"Edge ({i}, {j}) is part of the solution")
        print(f"Objective value: {m.objVal}")
        print(f"Sorted times (r): {[r[i].X for i in range(2)]}")
    else:
        print("No optimal solution found.")

    return m


def ex3_3_minOWA_regrets(edges, D_s1, D_s2, z1_opt, z2_opt, weights):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    m = gp.Model("minOWA_Regrets_3_3")

    x = m.addVars(G.edges, vtype=GRB.BINARY, name="x")
    r = m.addVars(2, vtype=GRB.CONTINUOUS, name="r") 
    r_ordered = m.addVars(2, vtype=GRB.CONTINUOUS, name="r_ordered")
    
    dest_node = max(G.nodes)

    m.addConstrs(
        (gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0
         for i in G.nodes if i not in [0, dest_node]), "flow")
    m.addConstr((gp.quicksum(x[j, dest_node] for j in G.predecessors(dest_node)) == 1), "sink")
    m.addConstr((gp.quicksum(x[0, j] for j in G.successors(0)) == 1), "source")

    m.addConstr(r[0] == z1_opt - gp.quicksum(D_s1[i, j] * x[i, j] for i, j in G.edges), "r_s1")
    m.addConstr(r[1] == z2_opt - gp.quicksum(D_s2[i, j] * x[i, j] for i, j in G.edges), "r_s2")

    m.addConstr(r_ordered[0] <= r[0])
    m.addConstr(r_ordered[0] <= r[1])
    m.addConstr(r_ordered[1] >= r[0])
    m.addConstr(r_ordered[1] >= r[1])
    m.addConstr(r_ordered[0] + r_ordered[1] == r[0] + r[1]) 

    m.setObjective(gp.quicksum(weights[i] * r_ordered[i] for i in range(2)), GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("\n\nOptimal solution for minOWA regrets:")
        for i, j in G.edges:
            if x[i, j].X > 0.5:
                print(f"Edge ({i}, {j}) is part of the solution")
        print("Objective value:", m.objVal)


def generate_graph_and_scenarios(p, density, n):
    G = nx.gnp_random_graph(p, density, directed=True)
    scenarios = {i: np.zeros((p, p)) for i in range(n)}
    for i in range(n):
        for (u, v) in G.edges():
            scenarios[i][u, v] = np.random.randint(1, 101)  
    return G, scenarios

def opt_each_scenario_takingG(G, D_s, dest_node):
    m = gp.Model()
    x = m.addVars(G.edges(), vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(D_s[u, v] * x[u, v] for u, v in G.edges()), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0
                  for i in G.nodes if i not in [0, dest_node]), "flow")
    m.addConstr(gp.quicksum(x[i, dest_node] for i in G.predecessors(dest_node)) == 1, "sink")
    m.addConstr(gp.quicksum(x[0, j] for j in G.successors(0)) == 1, "source")
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        solution_edges = [(i, j) for i, j in G.edges if x[i, j].X > 0.5]
        total_distance = sum(D_s[i, j] * x[i, j].X for i, j in G.edges)
        print(f"The optimal path: {solution_edges}")
        print(f"Total distance: {total_distance}")
        return m.objVal
    return float('inf')  # Return infinity if no solution is found


 

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

 
 
 
def resolution_time_graphs_for_each_np (n, p, num_instances):
    times = [] 
    for i in range(num_instances): 
        start_time = time.time()
        density = np.random.uniform(0.3, 0.5)
        G, scenarios = generate_graph_and_scenarios(p, density, n)
        dest_node = p - 1 
        optimal_distances = [opt_each_scenario_takingG(G, scenarios[scenario_ind], dest_node) for scenario_ind in scenarios]

        m = gp.Model("minimax_regret")
        x = m.addVars(G.edges(), vtype=GRB.BINARY, name="x")
        r = m.addVar(vtype=GRB.CONTINUOUS, name="r")


        m.setObjective(r, GRB.MINIMIZE)
        

        for scenario_index, D_s in scenarios.items():
            optimal_distance = optimal_distances[scenario_index]
            m.addConstr(r >= optimal_distance - gp.quicksum(D_s[u, v] * x[u, v] for u, v in G.edges()))

        m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0 
            for i in G.nodes if i not in [0, dest_node]),"flow")
        m.addConstr(gp.quicksum(x[0, j] for j in G.successors(0)) == 1, "source")
        m.addConstr(gp.quicksum(x[i, dest_node] for i in G.predecessors(dest_node)) == 1, "sink")

        m.optimize()
        if m.status == GRB.OPTIMAL:
            print(f"\n\n\nr is = {r.X}\n\n\n")
        elif m.status == GRB.INFEASIBLE:
            print("r is: Model is infeasible.")
        elif m.status == GRB.UNBOUNDED:
            print("r is: Model is unbounded.")

        end_time = time.time()
        times.append(end_time - start_time)

    average_time = np.mean(times)
    return average_time



def resolution_time_graphs_for_each_np_maximin (n, p, num_instances):
    times = [] 
    for i in range(num_instances): 
        start_time = time.time()
        density = np.random.uniform(0.3, 0.5)
        G, scenarios = generate_graph_and_scenarios(p, density, n)
        dest_node = p - 1 

        m = gp.Model("minimax_regret")
        x = m.addVars(G.edges(), vtype=GRB.BINARY, name="x")
        g = m.addVar(vtype=GRB.CONTINUOUS, name="g")


        m.setObjective(g, GRB.MAXIMIZE)
        

        for scenario_index, D_s in scenarios.items():
            m.addConstr(g <= gp.quicksum(D_s[u, v] * x[u, v] for u, v in G.edges()))

        m.addConstrs((gp.quicksum(x[j, i] for j in G.predecessors(i)) - gp.quicksum(x[i, j] for j in G.successors(i)) == 0 
            for i in G.nodes if i not in [0, dest_node]),"flow")
        m.addConstr(gp.quicksum(x[0, j] for j in G.successors(0)) == 1, "source")
        m.addConstr(gp.quicksum(x[i, dest_node] for i in G.predecessors(dest_node)) == 1, "sink")

        m.optimize()
        if m.status == GRB.OPTIMAL:
            print(f"\n\n\ng is = {g.X}\n\n\n")
        elif m.status == GRB.INFEASIBLE:
            print("g is: Model is infeasible.")
        elif m.status == GRB.UNBOUNDED:
            print("g is: Model is unbounded.")

        end_time = time.time()
        times.append(end_time - start_time)

    average_time = np.mean(times)
    return average_time



def exo3_4_checking_resolution_time_minimax_maximin():
    p_values = [10, 15, 20]
    n_values = [10, 5, 2]
    results_minimax = {}
    results_maximin = {}
    num_instances = 10
    for n in n_values:
        for p in p_values:
            average_time_minimax = resolution_time_graphs_for_each_np (n, p, num_instances)
            results_minimax[(n, p)] = average_time_minimax
            # average_time_maximin = resolution_time_graphs_for_each_np_maximin (n, p, num_instances)
            # results_maximin[(n, p)] = average_time_maximin
            
    print("\nSummary of Average Resolution Times:\n")
    result_list_minimax = []
    result_list_maximin = []
    
    
    for (n, p), avg_time in results_minimax.items():
        print(f"Scenarios: {n}, number of Nodes: {p} -> Avg Time: {avg_time:.4f} seconds")
        result_list_minimax.append(np.round(avg_time, 6))
    
    print(f'\nresults = {result_list_minimax}\n') 
    plot_heatmap(results_minimax, n_values, p_values) 
    
        
    # for (n, p), avg_time in results_maximin.items():
    #     print(f"Scenarios: {n}, number of Nodes: {p} -> Avg Time: {avg_time:.4f} seconds")
    #     result_list_maximin.append(np.round(avg_time, 6))

    # print(f'\nresults = {result_list_maximin}\n') 
    # plot_heatmap(results_maximin, n_values, p_values) 
    


def resolution_time_graphs_owa(n, p, num_instances, criterion):
    times = []

    for instance in range(num_instances):
        start_time = time.time()

        density = np.random.uniform(0.3, 0.5)
        G, scenarios = generate_graph_and_scenarios(p, density, n)
        dest_node = p - 1  # Dernier nœud comme destination
        edges = list(G.edges)

        if criterion == "maxOWA":
            ex3_3_maxOWA(edges, scenarios[0], scenarios[1], [2,1])

        elif criterion == "minOWA":
            z1_opt = opt_each_scenario_takingG(G, scenarios[0], dest_node)
            z2_opt = opt_each_scenario_takingG(G, scenarios[1], dest_node)
            ex3_3_minOWA_regrets(edges, scenarios[0], scenarios[1], z1_opt, z2_opt, [2,1])

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    return avg_time


def exo3_4_checking_resolution_times_owa():
    p_values = [10, 15, 20]
    n_values = [10, 5, 2]
    num_instances = 10
    results = {"maxOWA": {}, "minOWA": {}}

    for n in n_values:
        for p in p_values:
            avg_time_maxowa = resolution_time_graphs_owa(n, p, num_instances, "maxOWA")
            results["maxOWA"][(n, p)] = avg_time_maxowa

            avg_time_minowa = resolution_time_graphs_owa(n, p, num_instances, "minOWA")
            results["minOWA"][(n, p)] = avg_time_minowa
            
    plot_heatmap(results["maxOWA"], n_values, p_values)
    plot_heatmap(results["minOWA"], n_values, p_values)

    print("\nSummary of Average Resolution Times :")
    for criterion in ["maxOWA", "minOWA"]:
        print(f"\nCritere : {criterion}")
        for (n, p), avg_time in results[criterion].items():
            print(f"Scenarios : {n}, Number of nodes : {p} -> Avg time : {avg_time:.4f} seconds")





def run_question_3():
    
    # For the left graph:
    opt_z_1_left_graph = ex3_1_opt_for_each_senario(edges = EDGES_LEFT_GRAPH, D_s = D_S1_LEFT_GRAPH, scenario = 1, graph = "left", dest_node = 5)  # question 3.2
    opt_z_2_left_graph = ex3_1_opt_for_each_senario(edges = EDGES_LEFT_GRAPH, D_s = D_S2_LEFT_GRAPH, scenario = 2, graph = "left", dest_node = 5 )  # question 3.2 

    ex3_2_maximin(edges = EDGES_LEFT_GRAPH, D_s1 = D_S1_LEFT_GRAPH, D_s2 = D_S2_LEFT_GRAPH, dest_node = 5)  # question 3.3: Minimax Approach
    
    ex3_2_minimax_regret(edges = EDGES_LEFT_GRAPH, D_s1 = D_S1_LEFT_GRAPH, D_s2 = D_S2_LEFT_GRAPH, total_t_scenario1 = opt_z_1_left_graph , total_t_scenario2 = opt_z_2_left_graph, dest_node =5)  # question 3.3: Maximin R Approach
    
    ex3_3_maxOWA(EDGES_LEFT_GRAPH, D_S1_LEFT_GRAPH, D_S2_LEFT_GRAPH, [2,1])
    # ex3_3_minOWA_regrets(EDGES_LEFT_GRAPH, D_S1_LEFT_GRAPH, D_S2_LEFT_GRAPH, opt_z_1_left_graph, opt_z_2_left_graph, [2,1])
    
    
    # For the right graph:
    total_t_scenario1_right_graph = ex3_1_opt_for_each_senario(edges = EDGES_RIGHT_GRAPH, D_s = D_S1_RIGHT_GRAPH, scenario = 1, graph = "right", dest_node = 6 )  # question 3.2
    total_t_scenario2_right_graph = ex3_1_opt_for_each_senario(edges = EDGES_RIGHT_GRAPH, D_s = D_S2_RIGHT_GRAPH, scenario = 2, graph = "right", dest_node = 6)  # question 3.2
    
    ex3_2_maximin(edges = EDGES_RIGHT_GRAPH, D_s1 = D_S1_RIGHT_GRAPH, D_s2 = D_S2_RIGHT_GRAPH, dest_node = 6)  # question 3.3: Minimax Approach
    
    ex3_2_minimax_regret(edges = EDGES_RIGHT_GRAPH, D_s1 = D_S1_RIGHT_GRAPH, D_s2 = D_S2_RIGHT_GRAPH, total_t_scenario1 = total_t_scenario1_right_graph , total_t_scenario2 = total_t_scenario2_right_graph, dest_node = 6)  # question 3.3: Maximin R Approach
    
    ex3_3_maxOWA(EDGES_RIGHT_GRAPH, D_S1_RIGHT_GRAPH, D_S2_RIGHT_GRAPH, [2,1])
    # ex3_3_minOWA_regrets(EDGES_RIGHT_GRAPH, D_S1_RIGHT_GRAPH, D_S2_RIGHT_GRAPH, total_t_scenario1_right_graph, total_t_scenario2_right_graph, [2,1])
    
    # # Resolution Time for Minmax and Maximin Regret:
    # exo3_4_checking_resolution_time_minimax_maximin()
    
    # exo3_4_checking_resolution_times_owa()
