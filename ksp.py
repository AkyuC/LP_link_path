from typing import List
import copy
import heapq as hq
from graph import graph_class


def dijkstra(gp: graph_class, node_start, node_end, node_set: set = None):
    distances = dict()
    previous = dict()
    visited = set()
    path = list()

    for v in gp.graph:
        distances[v] = gp.INFINITY
        previous[v] = None

    if node_set is not None:
        visited = copy.deepcopy(node_set)
    visited.add(node_start)

    distances[node_start] = 0
    for node_adj in gp.graph[node_start]:
        previous[node_adj] = node_start
        distances[node_adj] = gp.graph[node_start][node_adj]

    node_min_weight = gp.INFINITY - 1
    node_min = node_start

    while True:
        for node in distances:
            if node not in visited and distances[node] < node_min_weight:
                node_min = node
                node_min_weight = distances[node]

        if node_min == node_end:
            path_weight = 0
            while node_min != node_start:
                path.append(node_min)
                path_weight += gp.graph[node_min][previous[node_min]]
                node_min = previous[node_min]
            path.append(node_start)
            path.reverse()
            return path, path_weight

        if node_min in visited:
            break

        visited.add(node_min)
        for node_adj in gp.graph[node_min]:
            if distances[node_min] + gp.graph[node_min][node_adj] < distances[node_adj]:
                previous[node_adj] = node_min
                distances[node_adj] = distances[node_min] + gp.graph[node_min][node_adj]

        node_min_weight = gp.INFINITY - 1

    return [], 0


def ksp(gp: graph_class, node_start, node_end, max_k):
    # https://en.wikipedia.org/wiki/Yen%27s_algorithm
    if max_k < 1:
        return {}

    paths = dict()
    paths_set = set()
    paths_weight = dict()
    paths_deviate = list()

    path_tmp, path_weight_tmp = dijkstra(gp, node_start, node_end)
    hq.heappush(paths_deviate, (path_weight_tmp, path_tmp))
    paths_set.add(tuple(path_tmp))

    num = 0
    while paths_deviate:
        paths_weight[num], paths[num] = hq.heappop(paths_deviate)
        paths_set.remove(tuple(paths[num]))
        path_now = paths[num]
        num += 1

        if num >= max_k:
            break

        for index in range(len(path_now) - 1):
            gp_tmp: graph_class = copy.deepcopy(gp)
            for path in paths:
                if paths[path][:index + 1] == path_now[:index + 1]:
                    each_path: List = paths[path]
                    if each_path[index + 1] in gp_tmp.graph[each_path[index]]:
                        del gp_tmp.graph[each_path[index]][each_path[index + 1]]
                        del gp_tmp.graph[each_path[index + 1]][each_path[index]]

            path_tmp, path_weight_tmp = dijkstra(gp_tmp, path_now[index], node_end, set(path_now[:index + 1]))
            if path_weight_tmp == 0:
                continue
            path_s2t_tmp = path_now[:index + 1] + path_tmp[1:]

            if tuple(path_s2t_tmp) not in paths_set:
                hq.heappush(paths_deviate, (path_weight_tmp, path_s2t_tmp))
                paths_set.add(tuple(path_s2t_tmp))

    return paths
