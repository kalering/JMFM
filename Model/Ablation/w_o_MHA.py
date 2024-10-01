import torch
import torch.nn as nn

class w_o_MHA(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, towers_hidden,num_tasks,level):
        super(w_o_MHA, self).__init__()
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.num_tasks = num_tasks
        self.level = level
        self.experts_shared = nn.ModuleList([nn.ModuleList() for _ in range(level)])


        for num in range(level):
            if num == 0:
                self.experts_shared[num].extend(nn.ModuleList(
                    [Expert(self.input_size, self.experts_out[num], self.experts_hidden) for i in
                     range(self.num_shared_experts)]))
            else:
                self.experts_shared[num].extend(nn.ModuleList(
                    [Expert(self.experts_out[num-1], self.experts_out[num], self.experts_hidden) for i in
                     range(self.num_shared_experts)]))

        self.experts_tasks = nn.ModuleList([nn.ModuleList() for _ in range(level)])
        self.dnns = nn.ModuleList([nn.ModuleList() for _ in range(level)])
        self.towers = nn.ModuleList()

        for num in range(level):
            self.experts_tasks[num] = nn.ModuleList()
            self.dnns[num] = nn.ModuleList()
            if num == 0:
                for i in range(self.num_tasks):
                    self.experts_tasks[num].append(nn.ModuleList(
                        [Expert(self.input_size, self.experts_out[num], self.experts_hidden) for j in
                         range(self.num_specific_experts)]))
                    self.dnns[num].append(
                        nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                      nn.Softmax()))
                self.dnns[num].append(
                        nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                      nn.Softmax()))
            elif num < level - 1:
                for i in range(self.num_tasks):
                    self.experts_tasks[num].append(nn.ModuleList(
                        [Expert(self.experts_out[num - 1], self.experts_out[num], self.experts_hidden) for j in
                         range(self.num_specific_experts)]))
                    self.dnns[num].append(
                        nn.Sequential(nn.Linear(self.experts_out[num-1], self.num_specific_experts + self.num_shared_experts),
                                      nn.Softmax()))
                self.dnns[num].append(
                    nn.Sequential(
                        nn.Linear(self.experts_out[num - 1], self.num_specific_experts + self.num_shared_experts),
                        nn.Softmax()))

            if num == level - 1 and num != 0:
                for i in range(self.num_tasks):
                    self.experts_tasks[num].append(nn.ModuleList(
                        [Expert(self.experts_out[num - 1], self.experts_out[num], self.experts_hidden) for j in
                         range(self.num_specific_experts)]))
                    self.dnns[num].append(
                        nn.Sequential(nn.Linear(self.experts_out[num - 1],
                                                self.num_specific_experts + self.num_shared_experts),
                                      nn.Softmax()))
        if level == 1:
            for i in range(self.num_tasks):
                self.towers.append(Tower(self.experts_out[0], 1, self.towers_hidden))
        else:
            for i in range(self.num_tasks):
                self.towers.append(Tower(self.experts_out[-1], 1, self.towers_hidden))

        self.soft = nn.Softmax(dim=1)

    def ple_net(self,x,level):
        experts_shared_o = [e(x[self.num_tasks]) for e in self.experts_shared[level]]
        experts_shared_o = torch.stack(experts_shared_o)

        experts_shared_x_list = []
        prev_size = 0
        for i in range(self.num_tasks):
            size = x[i].size(0)
            experts_shared_x_list.append(experts_shared_o[:, prev_size:prev_size + size, :])
            prev_size += size

        experts_task_outputs = []
        for i, x_task in enumerate(x):
            if i == len(x) - 1:
                break
            task_output = [e(x_task) for e in self.experts_tasks[level][i]]
            task_output = torch.stack(task_output)
            experts_task_outputs.append(task_output)
        level_outputs = []
        for i in range(self.num_tasks):
            selected = self.dnns[level][i](x[i])
            gate_expert_output = torch.cat((experts_task_outputs[i], experts_shared_x_list[i]), dim=0)
            gate_out = torch.einsum('abc, ba -> bc', gate_expert_output, selected)
            level_outputs.append(gate_out)

        selected = self.dnns[level][self.num_tasks](x[self.num_tasks])
        gate_expert_task = torch.cat([experts_task_outputs[k] for k in range(len(experts_task_outputs))], dim=1)
        gate_expert = torch.cat((gate_expert_task, experts_shared_o), dim=0)
        gate_expert_output = torch.einsum('abc, ba -> bc', gate_expert, selected)
        level_outputs.append(gate_expert_output)

        return level_outputs

    def cgc_net(self, x, level):
        experts_shared_o = [e(x[self.num_tasks]) for e in self.experts_shared[level]]
        experts_shared_o = torch.stack(experts_shared_o)

        experts_shared_x_list = []
        prev_size = 0
        for i in range(self.num_tasks):
            size = x[i].size(0)
            experts_shared_x_list.append(experts_shared_o[:, prev_size:prev_size + size, :])
            prev_size += size

        experts_task_outputs = []
        for i, x_task in enumerate(x):
            if i == len(x) - 1:
                break
            task_output = [e(x_task) for e in self.experts_tasks[level][i]]
            task_output = torch.stack(task_output)
            experts_task_outputs.append(task_output)
        level_outputs = []
        for i in range(self.num_tasks):
            selected = self.dnns[level][i](x[i])
            gate_expert_output = torch.cat((experts_task_outputs[i], experts_shared_x_list[i]), dim=0)
            gate_out = torch.einsum('abc, ba -> bc', gate_expert_output, selected)
            level_outputs.append(gate_out)

        return level_outputs

    def forward(self, x):
        ple_outputs = []
        data = x
        for i in range(self.level):
            if i == self.level - 1:
                ple_outputs = self.cgc_net(x=data,level=i)
            else:
                ple_outputs = self.ple_net(x=data,level=i)
                data = ple_outputs
        final_outputs = []
        for i in range(self.num_tasks):
            final_output = self.towers[i](ple_outputs[i])
            final_outputs.append(final_output)

        return final_outputs
