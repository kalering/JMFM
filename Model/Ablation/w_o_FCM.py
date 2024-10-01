import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class w_o_FCM(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, towers_hidden,
                 num_tasks, level):
        super(w_o_FCM, self).__init__()
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.num_tasks = num_tasks
        self.level = level
        self.experts_shared = nn.ModuleList(
            [self._create_experts_layer(num, self.input_size, experts_out, experts_hidden, True) for num in
             range(level)])
        self.experts_tasks = nn.ModuleList()
        for num in range(level):
            task_experts = nn.ModuleList(
                [self._create_experts_layer(num, self.input_size, experts_out, experts_hidden, False) for _ in
                 range(num_tasks)])
            self.experts_tasks.append(task_experts)
        self.dnns = nn.ModuleList([self._create_dnn_layer(num, self.input_size, self.experts_out,
                                                          self.num_specific_experts, self.num_shared_experts) for num in
                                   range(self.level)])
        self.towers = nn.ModuleList([Tower(self.experts_out[-1], 1, self.towers_hidden) for _ in range(self.num_tasks)])
        self.soft = nn.Softmax(dim=1)
        num_heads = 2
        self.attention_model = nn.ModuleList()
        for num in range(level):
            in_size = self.input_size if num == 0 else self.experts_out[num - 1]
            attention_module = MultiHeadAttention(in_size, in_size, num_heads)
            self.attention_model.append(attention_module)

    def _create_experts_layer(self, num, input_size, experts_out, experts_hidden, is_shared):
        layer = nn.ModuleList()
        in_features = input_size if num == 0 else experts_out[num - 1]
        out_features = experts_out[num]
        num_experts = self.num_shared_experts if is_shared else self.num_specific_experts

        for _ in range(num_experts):
            layer.append(Expert(in_features, out_features, experts_hidden))
        return layer

    def _create_dnn_layer(self, num, input_size, experts_out, num_specific_experts, num_shared_experts):
        dnn_layer = nn.ModuleList()
        in_features = input_size if num == 0 else experts_out[num - 1]
        num_outputs = num_specific_experts + num_shared_experts

        for _ in range(self.num_tasks + 1):
            dnn_layer.append(nn.Sequential(nn.Linear(in_features, num_outputs), nn.Softmax()))
        return dnn_layer

    def _process_experts_output(self, x, level):
        experts_task_outputs = []
        multi_head_attention_outputs = []

        for i, x_task in enumerate(x[:-1]):
            task_output = [e(x_task) for e in self.experts_tasks[level][i]]
            task_output = torch.stack(task_output)
            experts_task_outputs.append(task_output)

            x_task_prime = self.attention_model[level](x_task, x[-1], x[-1])
            multi_head_attention_outputs.append(x_task_prime)

        x[-1] = torch.cat(multi_head_attention_outputs, dim=0)

        experts_shared_o = [e(x[-1]) for e in self.experts_shared[level]]
        experts_shared_o = torch.stack(experts_shared_o)

        experts_shared_x_list = []
        prev_size = 0
        for i in range(self.num_tasks):
            size = x[i].size(0)
            experts_shared_x_list.append(experts_shared_o[:, prev_size:prev_size + size, :])
            prev_size += size

        return experts_task_outputs, experts_shared_o, experts_shared_x_list

    def _process_gate_out(self, task_output, selected_feature, selected):
        gate_expert_output = torch.cat((task_output, selected_feature), dim=0)
        return torch.einsum('abc, ba -> bc', gate_expert_output, selected)

    def ple_net(self, x, level):
        experts_task_outputs, experts_shared_o, experts_shared_x_list = self._process_experts_output(x,level)

        level_outputs = [
            self._process_gate_out(experts_task_outputs[i], experts_shared_x_list[i], self.dnns[level][i](x[i])) for i
            in
            range(self.num_tasks)]

        if level != self.level - 1:
            selected = self.dnns[level][self.num_tasks](x[-1])
            gate_expert_task = torch.cat(experts_task_outputs, dim=1)
            gate_expert = torch.cat((gate_expert_task, experts_shared_o), dim=0)
            level_outputs.append(torch.einsum('abc, ba -> bc', gate_expert, selected))

        return level_outputs

    def cgc_net(self, x, level):
        experts_task_outputs, experts_shared_o, experts_shared_x_list = self._process_experts_output(x,level)

        return [self._process_gate_out(experts_task_outputs[i], experts_shared_x_list[i], self.dnns[level][i](x[i])) for
                i
                in range(self.num_tasks)]

    def forward(self, x):
        cgc_outputs = []
        data = x
        for i in range(self.level):
            if i == self.level - 1:
                cgc_outputs = self.cgc_net(x=data,level=i)
            else:
                ple_outputs = self.ple_net(x=data,level=i)
                data = ple_outputs
        final_outputs = [self.towers[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        return final_outputs