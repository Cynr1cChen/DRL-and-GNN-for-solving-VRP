import os
import time
import torch
import numpy as np
from collections import OrderedDict, namedtuple
from itertools import product
from VRP.PPO_Agent import Agentppo, Memory
from VRP.creat_data import creat_data, reward1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_nodes = 101


def rollout(model, dataset, batch_size, steps):
    """
    使用指定的model在验证集上评估并返回平均cost。
    """
    model.eval()
    def eval_batch(bat):
        with torch.no_grad():
            cost, _ = model.act(bat, 0, steps, batch_size, True, False)
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()
    total_cost = torch.cat([eval_batch(bat.to(device)) for bat in dataset], 0)
    return total_cost

def run_ppo_training(steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim,
                     hidden_edge_dim, epoch, batch_size, conv_laysers, entropy_value,
                     eps_clip, timestep, ppo_epoch, data_loader, valid_loader):

    # 初始化记忆与智能体
    memory = Memory()
    agent = Agentppo(steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim,
                     hidden_edge_dim, ppo_epoch, batch_size, conv_laysers, entropy_value, eps_clip)
    agent.old_polic.to(device)

    folder = 'vrp-{}-GAT'.format(n_nodes)
    filename = '20201125'
    filepath = os.path.join(folder, filename)

    costs = []
    update_timestep = timestep

    for i in range(epoch):
        print('old_epoch:', i, '***************************************')
        agent.old_polic.train()
        times = []
        rewards2 = []
        epoch_start = time.time()
        start = epoch_start

        # 为每个epoch重新创建内存，以确保update间隔的数据收集
        memory.def_memory()

        for batch_idx, batch in enumerate(data_loader):
            x, attr, capcity, demand = batch.x, batch.edge_attr, batch.capcity, batch.demand
            x = x.view(batch_size, n_nodes, 2)
            attr = attr.view(batch_size, n_nodes*n_nodes, 1)
            capcity = capcity.view(batch_size, 1)
            demand = demand.view(batch_size, n_nodes, 1)
            batch = batch.to(device)

            actions, log_p = agent.old_polic.act(batch, 0, steps, batch_size, greedy, False)
            rewards = reward1(batch.x, actions, n_nodes)

            actions = actions.cpu().detach()
            log_p = log_p.cpu().detach()
            rewards = rewards.cpu().detach()

            # 将此批数据存入memory
            for i_batch in range(batch_size):
                memory.input_x.append(x[i_batch])
                memory.input_attr.append(attr[i_batch])
                memory.actions.append(actions[i_batch])
                memory.log_probs.append(log_p[i_batch])
                memory.rewards.append(rewards[i_batch])
                memory.capcity.append(capcity[i_batch])
                memory.demand.append(demand[i_batch])

            # 达到更新步长则进行PPO参数更新
            if (batch_idx + 1) % update_timestep == 0:
                agent.update(memory, i)
                memory.def_memory()  # 清空内存，为下一轮数据收集做准备

            rewards2.append(torch.mean(rewards).item())
            time_space = 100
            if (batch_idx + 1) % time_space == 0:
                end = time.time()
                times.append(end - start)
                start = end
                mean_reward = np.mean(rewards2[-time_space:])
                print('  Batch %d/%d, reward: %2.3f,took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_reward, times[-1]))

        # 在验证集上评估
        cost = rollout(agent.policy, valid_loader, batch_size, steps).mean()
        costs.append(cost.item())
        print('Problem:TSP%s / Average distance:' % n_nodes, cost.item())
        print(costs)

        epoch_dir = os.path.join(filepath, '%s' % i)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(agent.old_polic.state_dict(), save_path)

def train():
    class RunBuilder:
        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    params = OrderedDict(
        lr=[3e-4],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        epoch=[100],
        batch_size=[512],
        conv_laysers=[4],
        entropy_value=[0.01],
        eps_clip=[0.2],
        timestep=[1],
        ppo_epoch=[3],
        data_size=[512000],
        valid_size=[10000]
    )
    runs = RunBuilder.get_runs(params)

    from VRP.creat_data import creat_data
    for (lr, hidden_node_dim, hidden_edge_dim, epoch, batch_size, conv_laysers,
         entropy_value, eps_clip, timestep, ppo_epoch, data_size, valid_size) in runs:
        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'conv_laysers', 'epoch,batch_size',
              'entropy_value', 'eps_clip', 'timestep:','data_size','valid_size',
              lr, hidden_node_dim, hidden_edge_dim, epoch, batch_size,
              conv_laysers, entropy_value, eps_clip, timestep, data_size, valid_size)

        data_loader = creat_data(n_nodes, data_size, batch_size)
        valid_loader = creat_data(n_nodes, valid_size, batch_size)
        print('DATA CREATED/Problem size:', n_nodes)

        # 不再使用类，而是直接调用函数
        run_ppo_training(steps=n_nodes*2, greedy=False, lr=lr,
                         input_node_dim=3, hidden_node_dim=hidden_node_dim,
                         input_edge_dim=1, hidden_edge_dim=hidden_edge_dim,
                         epoch=epoch, batch_size=batch_size,
                         conv_laysers=conv_laysers,
                         entropy_value=entropy_value,
                         eps_clip=eps_clip,
                         timestep=timestep, ppo_epoch=ppo_epoch,
                         data_loader=data_loader, valid_loader=valid_loader)

if __name__ == "__main__":
    train()
