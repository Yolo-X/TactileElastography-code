import elastic_body_module
import numpy as np
import meshio
import scipy.optimize as sp
import time
def read_mesh(filename):
    stick = meshio.read(filename)
    nodes = stick.points
    cells = stick.cells_dict['tetra']
    print("load mesh from %s: %d nodes, %d cells"%(filename,len(nodes),len(cells)))
    return nodes,cells

def read_vector_from_file(filename):
    try:
        vector = np.loadtxt(filename)
        return vector
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def save_as_eigen_format(nodes, cells, nodes_filename="nodes.txt", cells_filename="cells.txt"):
    # save nodes
    np.savetxt(nodes_filename, nodes, fmt="%.6f")
    print(f"Nodes saved to {nodes_filename}")

    # save cells
    np.savetxt(cells_filename, cells, fmt="%d")
    print(f"Cells saved to {cells_filename}")


def compute(z_times, model,iter,lamda =1):
    #initialization 
    E = 0.2e6; nv = 0.49
    el = 0.001e6; eu = 100e6
    save_to_path = "results/"+ model + "/"

    cnt = 1
    
    temp = z_times + cnt - 1
    nodes,cells=read_mesh("data/" + model +".msh")
    
    for node in nodes:
        node[0] = lamda*node[0]
        node[1] = lamda*node[1]
        node[2] = lamda*node[2]
    save_as_eigen_format(nodes, cells)

    # create an object of ElasticBody
    elastic_body = elastic_body_module.ElasticBody(E, el, eu, z_times, nv)
    s = ((E-eu)/(el-eu))*np.ones(len(cells))
    if iter > 0:
        filename = save_to_path + 'we.txt'
        s = (read_vector_from_file(filename) - eu*np.ones(len(cells)))/(el - eu)
    list = []
    for i in range(len(cells)):
        list.append((0,1))
    
    elastic_body.load_data(cnt, "data-sampling/"+model)
    
    result = sp.minimize(elastic_body.eqn, s, jac= elastic_body.jac,method='TNC' , bounds = list, 
                         options={'factr': 1e2,
                                  'maxfun': 500, 
                                  'xtol': 1e-100,
                                  'ftol': 1e-100,
                                  'gtol': 1e0,
                                  'disp': True})

    
    due = elastic_body.due_sum
    doe = np.max(abs(due), axis=0)# reference for sensitivity values
    print(result)
    s= result.x
    J = result.jac
    we = el*s + eu*(np.ones((len(cells))) - s)
    loss = result.fun
    #reference for young's modulus
    np.savetxt(save_to_path + 'we.txt',we)

    return result.status, we, loss


def save_we_to_txt(we, file_path):
    with open(file_path, 'w') as f:
        for we_i in we:
            f.write(' '.join(map(str, we_i)) + '\n')

def load_we_from_txt(file_path):
    we = []
    with open(file_path, 'r') as f:
        for line in f:
            we_i = list(map(float, line.strip().split()))
            we.append(we_i)
    return we


def final_compute(sample_times, model, it_times = 0,lamda = 1, try_times = 1):
    
    we = []
    loss = []
    for i in range(it_times):
        results, we_i, loss_i = compute(sample_times, model,i,lamda)
        we.append(we_i)
        loss.append(loss_i)
        if results == 0:
            print(results)
            break
    #np.savetxt("results/" + model+"/loss%i-iter-%i.txt"%(sample_times, try_times), loss)
    #save_we_to_txt(we, "results/" + model+"/we%i-iter-%i.txt"%(sample_times, try_times))
    
    return

if __name__ == "__main__":
    
    try_times = 1
    rep_times = 1
    time_sum = []
    sp_times = 12 #sampling times
    # for example,we compute the distribution of model named "box-3"
    # force%i.txt, pnt%i.txt, A%i.txt, B%i.txt are needed before inverse computation (sampling data in forward simulation)
    time_start = time.time() 
    model = "box-3"; dt = 0
    final_compute(sp_times, model,it_times=rep_times, try_times = try_times)
    time_end = time.time() 
    time_sum.append(time_end - time_start + dt)


    print(time_sum)
