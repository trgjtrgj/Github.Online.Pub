import pandas as pd
# import sympy as smp
import matplotlib.pyplot as plt
import numpy as np

nodes = []
pipes = []

node_data_filename = "Node_Data.xlsx"
pipe_data_filename = "Pipe_Data.xlsx"
head_guesses_filename = "Head_Guesses.xlsx"

kinematic_viscosity = 13.57e-6
natural_gas_dnsity = 0.79
air_density = 1.225

def printM(M, rnd=4):
    max_col_num = [0 for _ in range(len(M[0]))]
    for j in range(len(M[0])):
        for i in range(len(M)):
            if len(str(round(M[i][j],rnd))) >= max_col_num[j]:
                max_col_num[j] = len(str(round(M[i][j],rnd)))
    spaces_num = sum(max_col_num) + len(max_col_num)*2 + 2
    spaces = ""
    for i in range(spaces_num):
        spaces += " "
    
    print("┌" + spaces + "┐") #┌└┐┘

    for i in range(len(M)):
        print("│  ", end="")
        for j in range(len(M[0])):
            print(round(M[i][j],rnd), end="")
            for k in range(max_col_num[j] - len(str(round(M[i][j],rnd))) + 2):
                print(' ', end="")
        print("│")
    print("└" + spaces + "┘") #┌└┐┘
    # try:
    #     for row in M:
    #         print(row)
    # except:
    #     raise ValueError

def h_terms(k, pi, pj):
    return (pj-pi)/abs(pj-pi)*np.sqrt(abs(pi-pj)/k)

def dh_terms(k, pi, pj):
    return 0.5/(np.sqrt(k*abs(pi-pj)))

class Node:
    def __init__(self, id, name, x, y, z, Q, Po):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.Q = Q
        self.Po = Po

class Pipe:
    def __init__(self, id, name, roughness, diameter, length, coordinates_1, coordinates_2):
        self.id = id
        self.name = name
        self.roughness = roughness
        self.diameter = diameter
        self.length = length
        self.Q = 0
        self.coordinates_1 = coordinates_1
        self.coordinates_2 = coordinates_2
        self.la = 0
        self.k = 0
        self.Re = 0
        self.rel_roughness = 0
        
    def calculate_lambda(self):
        if self.Re == 0:
            self.la = (1/(1.14 - 2*np.log10(self.rel_roughness)))**2
        else:
            self.la = (1/(1.14 - 2*np.log10(self.rel_roughness + 21.15/self.Re**0.9)))**2

    def calculate_Re(self):
        self.Re = self.Q*4/(np.pi*self.diameter)/kinematic_viscosity

    def calculate_rel_roughness(self):
        self.rel_roughness = self.roughness/self.diameter

    def calculate_k(self):
        self.k = self.la*self.length/self.diameter

# Reading an Excel file
df = pd.read_excel(pipe_data_filename)
num_pipes = len(df.iloc[:, 0].tolist())
for i in range(num_pipes):
    id, name, coordinates_1, coordinates_2, length, roughness, diameter = [df.iloc[:, j].tolist()[i] for j in range(7)]
    coordinates_1 = list(map(float, coordinates_1.split(',')))
    coordinates_2 = list(map(float, coordinates_2.split(',')))
    pipes.append(Pipe(id=id, name=name, coordinates_1=coordinates_1, coordinates_2=coordinates_2, length=length, roughness=roughness, diameter=diameter))

# print([pipe.diameter for pipe in pipes])
# exit(0)

df = pd.read_excel(node_data_filename)
num_nodes = len(df.iloc[:, 0].tolist())
for i in range(num_nodes):
    id, name, coordinates, Q, Po = [df.iloc[:, j].tolist()[i] for j in range(5)]
    x, y, z = map(float, coordinates.split(','))
    nodes.append(Node(id=id, name=name, x=x, y=y, z=z, Q=Q, Po=Po))

index_pipe_nei_nodes = [[] for _ in range(num_pipes)]
index_node_nei_pipes = [[] for _ in range(num_nodes)]
index_node_nei_nodes = [[] for _ in range(num_nodes)]

for pipe in pipes:
    pipe.calculate_rel_roughness()
    pipe.calculate_lambda()
    pipe.calculate_k()
    # print(pipe.k)
    counter = 0
    two_node_list = []
    for node in nodes:
        if counter >= 2:
            break
        if pipe.coordinates_1 == [node.x, node.y, node.z]:
            two_node_list.append(node)
            counter += 1
        
        elif pipe.coordinates_2 == [node.x, node.y, node.z]:
            two_node_list.append(node)
            counter += 1
    # print(pipe.id)
    index_pipe_nei_nodes[pipe.id] = two_node_list

# for pipe in pipes:
#     print(pipe.diameter)

# print([pipe.k for pipe in pipes])

# # print(index_pipe_nodes)
# exit(0)

for node in nodes:
    connected_pipes_list = []
    connected_nodes_list = []
    for pipe in pipes:
        if node in index_pipe_nei_nodes[pipe.id]:
            connected_pipes_list.append(pipe)
            connected_nodes_list.append([node_ for node_ in index_pipe_nei_nodes[pipe.id] if node_ != node][0])
    index_node_nei_pipes[node.id] = connected_pipes_list
    index_node_nei_nodes[node.id] = connected_nodes_list

# print(index_node_nei_pipes)
# print(index_node_nei_nodes)

# print(index_node_nei_nodes[2])
# print(index_pipe_nodes[0])
J = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
# matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
F = [0 for _ in range(num_nodes)]
# for i in range(num_nodes):
#     # print(len(index_node_nei_pipes[i]))
#     for j in range(len(index_node_nei_pipes[i])):
#         if index_pipe_nodes[j][0] == i:
#             print("FOUND")
#         if index_pipe_nodes[j][1] == i:
#             print("FOUND")

# for node in nodes:
#     print(node.Po)

# for node_id, node in enumerate(nodes):
#     for k in range(len(index_node_nei_pipes[node_id])):
#         print(f"k: {index_node_nei_pipes[node_id][k].k} ({type(index_node_nei_pipes[node_id][k].k)})")
#         print(f"node.Po: {node.Po} ({type(node.Po)})")
#         print(f"neighbor.Po: {index_node_nei_nodes[node_id][k].Po} ({type(index_node_nei_nodes[node_id][k].Po)})")
#         print()
#         print(d_Q(index_node_nei_pipes[node_id][k].k, node.Po, index_node_nei_nodes[node_id][k].Po))

# exit(0)

for node in nodes: #scan each node
    F[node.id] += node.Q
    for k in range(len(index_node_nei_pipes[node.id])): #scans each neighbor from the list of neighbours
        F[node.id] += h_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)
        J[node.id][index_node_nei_nodes[node.id][k].id] = dh_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)
        # matrix[node.id][index_node_nei_nodes[node.id][k].id] = 1
        # matrix[node.id][node.id] = 1
        J[node.id][node.id] -= dh_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)
        # print(-dh_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po))
        # print(d_Q(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po))
        # print(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)
        # J[index_node_nei_nodes[node.id][k].id][index_node_nei_pipes[node.id][k].id] = 1
        #print(index_node_nei_nodes[node.id][k].id) # Tells us the index of the columns in the matrix 
        #print(index_node_nei_pipes[node.id][k].id) #Tells us through which pipe the current node is connected with the neighbouring node we're checking
    # print("")


# printM(matrix)
# exit(0)
# printM(J, rnd=2)
# print(J)
# print(np.dot(-np.array(F), np.linalg.inv(np.array(J))))
# for row in J:
#     print(sum(row))

dh = np.dot(-np.array(F), np.linalg.inv(np.array(J)))



# for pipe in pipes:
#     (index_pipe_nei_nodes[pipe.id][0]-index_pipe_nei_nodes[pipe.id][1])
#     for k in range(len(index_node_nei_pipes[node.id])):
#         index_node_nei_pipes[node.id][k].id


iterations = 50
head_log = []
head_log_data = [[0 for _ in range(iterations)] for _ in range(num_nodes)]
    

for i in range(iterations):
    for node in nodes:
        node.Po += dh[node.id]
        # print(i, node.id)
        head_log_data[node.id][i] += node.Po
        head_log.append(round(node.Po,1))
    
    # print(head_log)   

    for pipe in pipes:
        pipe.calculate_rel_roughness()
        pipe.calculate_Re()
        pipe.calculate_lambda()
        pipe.calculate_k()
    # printM([[pipe.rel_roughness, pipe.Re, pipe.la, pipe.k]])

    for node in nodes:
        F[node.id] += node.Q
        for k in range(len(index_node_nei_pipes[node.id])):
            F[node.id] += h_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)
            J[node.id][index_node_nei_nodes[node.id][k].id] = dh_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)
            J[node.id][node.id] -= dh_terms(index_node_nei_pipes[node.id][k].k, node.Po, index_node_nei_nodes[node.id][k].Po)

    dh = np.dot(-np.array(F), np.linalg.inv(np.array(J)))
    printM([dh])

exit(0)

# print(head_log_data[0])
# for i in range(num_nodes):
#     plt.plot([j+1 for j in range(iterations)], head_log_data[i])

# plt.axvline()
# plt.axhline()
plt.title("Convergence")
plt.xlabel("Iterations")
plt.ylabel("Head Valaues")
plt.grid()
plt.legend()    
plt.show()

exit(0)

log_file_path = "C:/Users/trgjt/OneDrive - Εθνικό Μετσόβιο Πολυτεχνείο/Programming/PYTHON/Proj_2/log2.txt"  # Replace with your desired path

def file_printM(M, path, rnd=4):
    with open(path, 'w') as file:
        max_col_num = [0 for _ in range(len(M))]
        for j in range(len(M)):
            for i in range(len(M)):
                if len(str(round(M[i][j],rnd))) >= max_col_num[j]:
                    max_col_num[j] = len(str(round(M[i][j],rnd)))
        spaces_num = sum(max_col_num) + len(max_col_num)*2 + 2
        spaces = ""
        for i in range(spaces_num):
            spaces += " "
        
        file.write("┌" + spaces + "┐\n") #┌└┐┘

        for i in range(len(M)):
            file.write("│  ")
            for j in range(len(M)):
                file.write(str(round(M[i][j],rnd)))
                for k in range(max_col_num[j] - len(str(round(M[i][j],rnd))) + 2):
                    file.write(' ')
            file.write("│\n")
        file.write("└" + spaces + "┘\n")
    print(f"Matrix written to: {path}")

# file_printM(J, log_file_path)

npJ = np.array(J)
print("Cannot find inverse!") if np.linalg.det(npJ) == 0 else print(np.linalg.inv(npJ))
exit(0)

# print(J)