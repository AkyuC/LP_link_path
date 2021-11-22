import ksp
from graph import graph_class
import numpy as np
from scipy import optimize


def fun():
    def v(x):
        return x[-1]

    return v


def cons_eq(index_start, index_end, h):
    def v(x):
        s = 0
        for index in range(index_start, index_end):
            s += x[index]
        return s - h

    return v


def cons_ineq(capacity, index_list):
    def v(x):
        s = 0
        for index in index_list:
            s += x[index]
        return capacity * x[-1] - s

    return v


def cons_bounds(index):
    def v(x):
        return x[index]

    return v


if __name__ == "__main__":
    # conda config --set auto_activate_base false
    max_k = 5
    max_demand = 12
    gp = graph_class()

    paths = dict()
    demand_no = 0
    for src in gp.fl_demand:
        for dst in gp.fl_demand[src]:
            paths[(src, dst, demand_no)] = ksp.ksp(gp, src, dst, max_k)
            demand_no += 1

    edge2paths = dict()
    for src, dst, demand_no in paths:
        for path_no in paths[(src, dst, demand_no)]:
            path = paths[(src, dst, demand_no)][path_no]
            for path_index in range(len(path) - 1):
                if (path[path_index], path[path_index + 1]) not in edge2paths:
                    edge2paths[(path[path_index], path[path_index + 1])] = list()
                edge2paths[(path[path_index], path[path_index + 1])].append((src, dst, demand_no, path_no))

    x0 = [0 for i in range(max_demand*max_k)]
    x0.append(1)

    cons = list()

    for src, dst, demand_no in paths:
        cons.append({'type': 'eq', 'fun': cons_eq(demand_no * max_k, demand_no * max_k + max_k, gp.fl_demand[src][dst])})

    for node1, node2 in edge2paths:
        index_path = list()
        for src, dst, demand_no, path_no in edge2paths[(node1, node2)]:
            index_path.append(demand_no * max_k + path_no)
        if (node2, node1) in edge2paths:
            for src, dst, demand_no, path_no in edge2paths[(node1, node2)]:
                index_path.append(demand_no * max_k + path_no)
        if index_path is not None:
            cons.append({'type': 'ineq', 'fun': cons_ineq(gp.link_c[node1][node2], index_path)})

    for i in range(len(x0)):
        cons.append({'type': 'ineq', 'fun': cons_bounds(i)})

    x0 = np.asarray(tuple(x0))

    cons = tuple(cons)

    res = optimize.minimize(fun(), x0, method='SLSQP', constraints=cons)

    print(res.fun)
    print(res.success)
    print(res.x)
