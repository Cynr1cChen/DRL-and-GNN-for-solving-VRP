import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免弹出图形窗口

import torch
import os
import numpy as np
from torch_geometric.data import Data, Batch
from VRP.creat_data import reward1, creat_instance
from VRP.Actor_network import Model
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import matplotlib.pyplot as plt
import logging  # 添加日志模块

# 设置VRPy的日志级别为WARNING，抑制INFO信息
logging.getLogger('vrpy').setLevel(logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_optimal_solution(node_coords, demands, capacity):
    """使用 Google OR-Tools 计算最优解。"""
    # 创建数据
    data = {}
    data['locations'] = [(x, y) for x, y in node_coords]
    data['num_locations'] = len(data['locations'])
    data['depot'] = 0  # 假设第一个节点是仓库

    # 将坐标和需求缩放为整数
    scale_factor = 1000
    int_locations = [(int(x * scale_factor), int(y * scale_factor)) for x, y in data['locations']]

    # 缩放需求
    demand_scale_factor = 100
    int_demands = [int(d * demand_scale_factor) for d in demands]
    int_capacity = int(capacity * demand_scale_factor)

    # 估计车辆数量
    total_demand = sum(int_demands[1:])  # 排除仓库需求
    n_vehicles = int(np.ceil(total_demand / int_capacity))
    data['vehicle_capacities'] = [int_capacity] * n_vehicles
    data['num_vehicles'] = n_vehicles

    # 创建距离矩阵
    def compute_euclidean_distance_matrix(locations):
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    distances[from_counter][to_counter] = int(
                        np.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1])))
        return distances

    data['distance_matrix'] = compute_euclidean_distance_matrix(int_locations)

    # 创建路由索引管理器
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])

    # 创建路由模型
    routing = pywrapcp.RoutingModel(manager)

    # 创建并注册距离回调函数
    def distance_callback(from_index, to_index):
        """返回两个节点之间的距离。"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 设置每条边的成本
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 添加容量约束
    def demand_callback(from_index):
        """返回节点的需求。"""
        from_node = manager.IndexToNode(from_index)
        return int_demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # 无容量松弛
        data['vehicle_capacities'],  # 车辆最大容量
        True,  # 从零开始累计
        'Capacity')

    # 设置初始解启发式方法
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(60)  # 设置求解器时间限制

    # 求解问题
    solution = routing.SolveWithParameters(search_parameters)

    # 如果问题不可行或在时间限制内未找到解
    if solution is None:
        print("No solution found by OR-Tools.")
        return None, None

    # 获取路线和总距离
    total_distance = 0
    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route = [data['depot']]  # 从仓库开始
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        # 返回仓库
        last_node = manager.IndexToNode(previous_index)
        route.append(data['depot'])
        route_distance += data['distance_matrix'][last_node][data['depot']]
        total_distance += route_distance
        routes.append(route)

    # 将总距离转换回原始尺度
    total_distance = total_distance / scale_factor

    return total_distance, routes


def vrpy_solve_vrp(node_coords, demands, capacity):
    """使用 VRPy 求解 VRP 问题。"""
    import networkx as nx
    from vrpy import VehicleRoutingProblem

    # 确保 demands 和 capacity 为整数
    demands = demands.astype(int)
    capacity = max(1, int(capacity))

    n_nodes = len(node_coords)
    G = nx.DiGraph()

    # 添加 'Source' 和 'Sink' 节点，代表车辆的起点和终点
    G.add_node("Source", demand=0, x=node_coords[0][0], y=node_coords[0][1])
    G.add_node("Sink", demand=0, x=node_coords[0][0], y=node_coords[0][1])

    # 添加客户节点
    for i in range(1, n_nodes):
        G.add_node(
            i,
            demand=demands[i],
            x=node_coords[i][0],
            y=node_coords[i][1],
        )

    # 从 'Source' 到每个客户节点的边
    for i in range(1, n_nodes):
        distance = np.linalg.norm(np.array(node_coords[0]) - np.array(node_coords[i]))
        G.add_edge("Source", i, cost=distance)

    # 从客户节点到 'Sink' 的边
    for i in range(1, n_nodes):
        distance = np.linalg.norm(np.array(node_coords[i]) - np.array(node_coords[0]))
        G.add_edge(i, "Sink", cost=distance)

    # 客户节点之间的边
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                distance = np.linalg.norm(np.array(node_coords[i]) - np.array(node_coords[j]))
                G.add_edge(i, j, cost=distance)

    # 创建 VRP 实例
    prob = VehicleRoutingProblem(G, load_capacity=capacity)

    # 解决问题
    prob.solve(cspy=False, time_limit=60)

    # 提取最佳解
    best_routes = []
    for route in prob.best_routes.values():
        # 将节点名称转换为整数，并将 'Source' 和 'Sink' 替换为仓库节点 0
        processed_route = []
        for node in route:
            if node == "Source" or node == "Sink":
                processed_route.append(0)
            else:
                processed_route.append(int(node))
        best_routes.append(processed_route)

    best_total_distance = prob.best_value

    return best_total_distance, best_routes


def process_model_tour(tour_tensor, n_node):
    tour = tour_tensor.squeeze(0).numpy()
    if tour.ndim == 0 or tour.size == 0:
        print("Tour is scalar or empty, cannot process.")
        return []
    # 转换为整数
    tour = tour.astype(int)
    # 初始化路线列表
    routes = []
    current_route = []
    for node in tour:
        current_route.append(node)
        if node == 0:
            if len(current_route) > 1:
                # 如果当前路线不以0开始，添加起点0
                if current_route[0] != 0:
                    current_route = [0] + current_route
                routes.append(current_route)
            current_route = []
    # 处理最后一条未添加的路线
    if len(current_route) > 0:
        if current_route[0] != 0:
            current_route = [0] + current_route
        if current_route[-1] != 0:
            current_route.append(0)
        routes.append(current_route)
    return routes


def plot_routes_subplot(ax, node_coords, routes, title):
    x_coords = [coord[0] for coord in node_coords]
    y_coords = [coord[1] for coord in node_coords]
    ax.scatter(x_coords[0], y_coords[0], c='red', marker='s', label='Depot')
    ax.scatter(x_coords[1:], y_coords[1:], c='blue', label='Customers')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for idx, route in enumerate(routes):
        route_coords = [node_coords[node] for node in route]
        xs, ys = zip(*route_coords)
        ax.plot(xs, ys, c=colors[idx], label=f'Vehicle {idx + 1}')
    ax.set_title(title)
    ax.legend()


def test_main(n_node, num_experiments=3):
    datas = []

    if n_node == 21 or n_node == 51 or n_node == 101:
        node_ = np.loadtxt('./data_for_evaluate/{}nodes_test_data.csv'.format(n_node - 1), dtype=float, delimiter=',')
        demand_ = np.loadtxt('./data_for_evaluate/{}nodes_demand.csv'.format(n_node - 1), dtype=float, delimiter=',')
        capcity_ = np.loadtxt('./data_for_evaluate/{}nodes_capcity.csv'.format(n_node - 1), dtype=float, delimiter=',')
        batch_size = 1  # 使用批量大小为1，逐个处理实例
    else:
        print('Please enter 21, 51 or 101')
        return
    node_ = node_.reshape(-1, n_node, 2)
    demand_ = demand_.reshape(-1, n_node)
    capcity_ = capcity_.reshape(-1, 1)

    data_size = node_.shape[0]

    edges = np.zeros((data_size, n_node, n_node, 1))
    for k, data in enumerate(node_):
        for i, (x1, y1) in enumerate(data):
            for j, (x2, y2) in enumerate(data):
                d = np.hypot(x1 - x2, y1 - y2)
                edges[k][i][j][0] = d
    edges_ = edges.reshape(data_size, -1, 1)

    edges_index = []
    for i in range(n_node):
        for j in range(n_node):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index).transpose(0, 1)
    for i in range(data_size):
        data = Data(x=torch.from_numpy(node_[i]).float(), edge_index=edges_index,
                    edge_attr=torch.from_numpy(edges_[i]).float(),
                    demand=torch.tensor(demand_[i]).unsqueeze(-1).float(),
                    capcity=torch.tensor(capcity_[i]).unsqueeze(-1).float())
        datas.append(data)

    print('Data prepared')

    # 创建保存图像的文件夹
    output_dir = 'test_result_image'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 进行实验
    for exp_num in range(num_experiments):
        print(f'\n========== Experiment {exp_num + 1} ==========')
        # 为了可重复性，我们可以选择特定的索引或使用随机
        random_idx = np.random.randint(0, len(datas))
        random_instance = datas[random_idx]
        node_coords = random_instance.x.numpy()
        demands = random_instance.demand.numpy().flatten()
        capacity = random_instance.capcity.item()
        # 如果需要，缩放需求和容量到原始尺度
        demands *= 10
        capacity *= 10

        # 确保 demands 和 capacity 为整数
        demands = demands.astype(int)
        capacity = max(1, int(capacity))

        # 准备收集不同算法的路线
        routes_dict = {}
        costs_dict = {}

        # OR-Tools 求解器
        optimal_cost, optimal_routes = compute_optimal_solution(node_coords, demands, capacity)
        if optimal_cost is not None:
            print('OR-Tools solution distance: {:.4f}'.format(optimal_cost))
            routes_dict['OR-Tools'] = optimal_routes
            costs_dict['OR-Tools'] = optimal_cost
        else:
            print('Could not compute optimal solution with OR-Tools.')

        # VRPy 求解器
        vrpy_cost, vrpy_routes = vrpy_solve_vrp(node_coords, demands, capacity)
        if vrpy_cost is not None:
            print('VRPy solution distance: {:.4f}'.format(vrpy_cost))
            routes_dict['VRPy'] = vrpy_routes
            costs_dict['VRPy'] = vrpy_cost
        else:
            print('Could not compute solution with VRPy.')

        # 模型求解
        agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
        agent.to(device)

        filepath = os.path.join('trained_model', f'{n_node}')
        if os.path.exists(filepath):
            path1 = os.path.join(filepath, 'actor.pt')
            agent.load_state_dict(torch.load(path1, device))
        else:
            print("Model not found!")
            return
        agent.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([random_instance]).to(device)
            actions, log_p = agent(batch, n_node * 2, True)
            model_cost = reward1(batch.x, actions.detach(), n_node)
            model_cost = model_cost.item()
        # 处理模型路线
        model_routes = process_model_tour(actions.cpu(), n_node)
        routes_dict['Model'] = model_routes
        costs_dict['Model'] = model_cost

        print('Model solution distance: {:.4f}'.format(model_cost))

        # 输出模型的每辆车路线
        print("\nModel routes:")
        for vehicle_idx, route in enumerate(model_routes):
            print(f"Vehicle {vehicle_idx + 1}: {' -> '.join(map(str, route))}")

        # 绘制不同算法的路线
        num_algorithms = len(routes_dict)
        fig, axs = plt.subplots(1, num_algorithms, figsize=(5 * num_algorithms, 5))
        if num_algorithms == 1:
            axs = [axs]
        for idx, (algo_name, routes) in enumerate(routes_dict.items()):
            ax = axs[idx]
            plot_routes_subplot(ax, node_coords, routes, f'{algo_name}\nDistance: {costs_dict[algo_name]:.4f}')
        plt.suptitle(f'Experiment {exp_num + 1}')
        plt.tight_layout()

        # 构建保存图像的路径和名称
        image_filename = f'vrp{n_node}_experiment{exp_num + 1}.png'
        image_path = os.path.join(output_dir, image_filename)
        plt.savefig(image_path)  # 保存图像到文件
        plt.close()  # 关闭图像，释放内存
        print(f'Plot saved to {image_path}')

        # 比较结果
        print('\n--- Comparison of Results ---')
        if 'OR-Tools' in costs_dict:
            gap = (costs_dict['Model'] - costs_dict['OR-Tools']) / costs_dict['OR-Tools'] * 100
            print('Gap between Model and OR-Tools solution: {:.2f}%'.format(gap))
        if 'VRPy' in costs_dict:
            gap = (costs_dict['Model'] - costs_dict['VRPy']) / costs_dict['VRPy'] * 100
            print('Gap between Model and VRPy solution: {:.2f}%'.format(gap))
        print('====================================\n')


if __name__ == "__main__":
    node_counts = [21]  # 节点数量
    num_experiments = 1  # 每个节点实验次数

    for n_node in node_counts:
        print(f'\n====== Starting tests for VRP with {n_node} nodes ======')
        test_main(n_node, num_experiments=num_experiments)
        print(f'\n====== Completed tests for VRP with {n_node} nodes ======\n')
