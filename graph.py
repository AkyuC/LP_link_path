import random
import copy


class graph_class:
    INFINITY = 100000  # 无限大的边值

    def __init__(self) -> None:
        # 邻接矩阵
        self.graph = {
            1: {2: 1, 3: 1},
            2: {1: 1, 3: 1, 4: 1},
            3: {1: 1, 2: 1, 7: 1, 9: 1, 11: 1},
            4: {2: 1, 5: 1, 6: 1, 7: 1},
            5: {4: 1, 6: 1, 7: 1},
            6: {4: 1, 5: 1, 8: 1},
            7: {3: 1, 4: 1, 5: 1, 8: 1, 9: 1},
            8: {6: 1, 7: 1, 10: 1},
            9: {3: 1, 7: 1, 10: 1, 11: 1, 12: 1},
            10: {8: 1, 9: 1, 13: 1, 14: 1},
            11: {3: 1, 9: 1, 12: 1, 15: 1, 16: 1},
            12: {9: 1, 11: 1, 13: 1, 17: 1},
            13: {10: 1, 12: 1, 14: 1, 21: 1},
            14: {10: 1, 13: 1, 23: 1},
            15: {11: 1, 17: 1, 18: 1},
            16: {11: 1, 18: 1},
            17: {12: 1, 15: 1, 19: 1, 20: 1, 21: 1},
            18: {15: 1, 16: 1, 19: 1},
            19: {17: 1, 18: 1, 20: 1},
            20: {17: 1, 19: 1, 21: 1, 22: 1},
            21: {13: 1, 17: 1, 20: 1, 22: 1, 23: 1},
            22: {20: 1, 21: 1, 24: 1},
            23: {14: 1, 21: 1, 24: 1},
            24: {22: 1, 24: 1},
        }

        # 随机生成边权重
        for node in self.graph:
            for adj_node in self.graph[node]:
                self.graph[node][adj_node] = random.randint(1, 5)
                self.graph[adj_node][node] = self.graph[node][adj_node]

        # 随机生成边容量
        self.link_c = copy.deepcopy(self.graph)
        for node in self.graph:
            for adj_node in self.graph[node]:
                self.link_c[node][adj_node] = random.randint(400, 1000)
                self.link_c[adj_node][node] = self.link_c[node][adj_node]

        # 特定个数的需求，流量大小随机生成
        self.fl_demand = dict()
        for node in range(1, int(len(self.graph)/2)+1):
            self.fl_demand[node] = dict()
            # random_key = node
            # while random_key == node:
            #     random_key = random.choice(list(self.graph))
            self.fl_demand[node][node+int(len(self.graph)/2)] = random.randint(10, 20)
