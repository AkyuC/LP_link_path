import ksp
from graph import graph_class
from mip import *

if __name__ == "__main__":
    # conda config --set auto_activate_base false
    md = Model(solver_name=CBC)
    max_k = 5
    max_demand = 12
    gp = graph_class()

    paths = dict()
    x = dict()
    x2src_dst = dict()
    for src in gp.fl_demand:
        for dst in gp.fl_demand[src]:
            paths[(src, dst)] = ksp.ksp(gp, src, dst, max_k)
            for i in range(max_k):
                name = "n{}2n{}_p{}".format(src, dst, i + 1)
                x[name] = md.add_var(name=name, var_type='B')
                x2src_dst[name] = (src, dst, i)

    r = md.add_var(name="r", lb=0.0, ub=1.0)

    edge2paths = dict()
    for src, dst in paths:
        for path_no in paths[(src, dst)]:
            path = paths[(src, dst)][path_no]
            for path_index in range(len(path) - 1):
                if (path[path_index], path[path_index + 1]) not in edge2paths:
                    edge2paths[(path[path_index], path[path_index + 1])] = list()
                edge2paths[(path[path_index], path[path_index + 1])].append((src, dst, path_no))

    for src in gp.fl_demand:
        for dst in gp.fl_demand[src]:
            md.add_constr(xsum(x["n{}2n{}_p{}".format(src, dst, i + 1)] for i in range(max_k)) == 1)

    for node1, node2 in edge2paths:
        name_list = list()
        for src, dst, path_no in edge2paths[(node1, node2)]:
            name_list.append(("n{}2n{}_p{}".format(src, dst, path_no + 1), src, dst))
        if (node2, node1) in edge2paths:
            for src, dst, path_no in edge2paths[(node1, node2)]:
                name_list.append(("n{}2n{}_p{}".format(src, dst, path_no + 1), src, dst))
        if name_list is not None:
            md.add_constr(
                xsum(x[name[0]] * gp.fl_demand[name[1]][name[2]] for name in name_list) <= r * gp.link_c[node1][node2])

    # 定义目标函数： Maximize x + 10 * y
    md.objective = r

    md.max_gap = 0.05
    status = md.optimize(max_seconds=300)
    if status == OptimizationStatus.OPTIMAL:
        print('\noptimal solution cost {} found'.format(md.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('\nsol.cost {} found, best possible: {}'.format(md.objective_value, md.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('\nno feasible solution found, lower bound is: {}'.format(md.objective_bound))
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        print('\nsolution:')
        for v in md.vars:
            if abs(v.x) > 1e-6 and v.name != r.name:  # only printing non-zeros
                src, dst, path_no = x2src_dst[v.name]
                print('node{} - node{}, demand:{}, path:{}'.format(src, dst, gp.fl_demand[src][dst],
                                                                   paths[(src, dst)][path_no]))
