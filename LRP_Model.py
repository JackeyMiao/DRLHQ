import torch
import torch.nn as nn
import torch.nn.functional as F


class LRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_nodes = None
        self.encoded_nodes_mean = None
        # shape: (batch_size, node_size, embedding_dim)
        self.depot_size = None
        self.encoded_depots = None
        self.encoded_customers = None

        # LRP
        self.depot_mask = None
        self.depot_mask_backup = None
        self.prediction_martrix = None


    def pre_forward(self, reset_state):
        depot_x_y = reset_state.depot_x_y
        # shape: (batch_size, depot_size, 2)
        customer_x_y = reset_state.customer_x_y
        # shape: (batch_size, customer_size, 2)
        customer_demand = reset_state.customer_demand
        # shape: (batch_size, customer_size)
        self.depot_size = reset_state.depot_size
        self.customer_size = reset_state.customer_size
        depot_cost = reset_state.depot_cost
        depot_capacity = reset_state.depot_capacity
        depot_capacity = reset_state.depot_capacity

        depot_capacity_cost_ratio_origin = depot_capacity.clone() / depot_cost.clone()
        min_vals = depot_capacity_cost_ratio_origin.min(dim=1, keepdim=True).values
        max_vals = depot_capacity_cost_ratio_origin.max(dim=1, keepdim=True).values
        depot_capacity_cost_ratio = (depot_capacity_cost_ratio_origin - min_vals) / (max_vals - min_vals)
        depot_customer_capacity = depot_capacity.clone() / torch.sum(customer_demand,dim=1)[:,None]
        customer_x_y_demand = torch.cat((customer_x_y, customer_demand[:, :, None]), dim=2)
        depot_x_y_cost = torch.cat((depot_x_y, depot_capacity_cost_ratio[:, :, None], depot_customer_capacity[:, :, None]), dim=2)
        
        # shape: (batch_size, customer_size, 3)
        self.encoded_nodes = self.encoder(depot_x_y_cost, customer_x_y_demand)
        # self.encoded_nodes_mean = self.encoded_nodes.mean(dim=1,keepdim=True).repeat(1, mt_size, 1)
        self.encoded_nodes_mean = self.encoded_nodes.mean(dim=1,keepdim=True)
        # shape: (batch_size, node_size, embedding_dim)
        self.decoder.set_k_v(self.encoded_nodes, reset_state)
        


    def forward(self, state, last_hh):
        batch_size = state.batch_idx.size(0)
        mt_size = state.mt_idx.size(1)

        if state.selected_count == 1:  # second move mt
            selected = torch.arange(self.depot_size, self.depot_size + mt_size)[None, :].expand(batch_size, mt_size)
            probability = torch.ones(size=(batch_size, mt_size))
        else:
            prob, last_hh = self.decoder(self.encoded_nodes, self.encoded_nodes_mean, state, mask=state.mask, last_hh=last_hh)
            # shape: (batch_size, mt_size, node_size)
            if self.training or self.model_params['sample']:
                while True:
                    with torch.no_grad():
                        selected = prob.reshape(batch_size * mt_size, -1).multinomial(1).squeeze(dim=1) \
                            .reshape(batch_size, mt_size)
                        # shape: (batch_size, mt_size)
                    probability = prob[state.batch_idx, state.mt_idx, selected].reshape(batch_size, mt_size)
                    # shape: (batch_size, mt_size)
                    if (probability != 0).all():
                        break
            else:
                selected = prob.argmax(dim=2)
                # shape: (batch_size, mt_size)
                probability = None
        return selected, probability, last_hh
    
    def get_expand_prob(self, state, last_hh):
        prob, last_hh = self.decoder(self.encoded_nodes, self.encoded_nodes_mean, state, mask=state.mask, last_hh=last_hh)

        return prob, last_hh


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.encoder_layer_num = self.model_params['encoder_layer_num']
        self.depot_size = None
        self.customer_size = None

        self.embedding_depot = nn.Linear(4, self.embedding_dim)
        self.embedding_customer = nn.Linear(3, self.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])


    def forward(self, depot_x_y_cost_capacity, customer_x_y_demand):
        self.depot_size = depot_x_y_cost_capacity.size()[1]
        self.customer_size = customer_x_y_demand.size()[1]

        # depot_x_y shape: (batch_size, depot_size, 2)
        # customer_x_y_demand shape: (batch_size, customer_size, 3)
        embedded_depots = self.embedding_depot(depot_x_y_cost_capacity)
        # shape: (batch_size, depot_size, embedding_dim)
        embedded_customers = self.embedding_customer(customer_x_y_demand)
        # shape: (batch_size, customer_size, embedding_dim)
        embedded_nodes = torch.cat((embedded_depots, embedded_customers), dim=1)
        # shape: (batch_size, node_size, embedding_dim)
        for layer in self.layers:
            embedded_nodes = layer(embedded_nodes)


        return embedded_nodes
        # shape: (batch_size, node_size, embedding_dim)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.norm1 = Norm(**model_params)
        self.ff = FF(**model_params)
        self.norm2 = Norm(**model_params)

    def forward(self, out):

        # shape: (batch_size, node_size, embedding_dim)
        q = multi_head_qkv(self.Wq(out), head_num=self.head_num)
        k = multi_head_qkv(self.Wk(out), head_num=self.head_num)
        v = multi_head_qkv(self.Wv(out), head_num=self.head_num)
        # shape: (batch_size, head_num, node_size, qkv_dim)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch_size, node_size, head_num * qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch_size, node_size, embedding_dim)
        out1 = self.norm1(out, multi_head_out)
        out2 = self.ff(out1)
        out3 = self.norm2(out1, out2)
        return out3
        # shape :(batch_size, node_size, embedding_dim)


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']
        self.clip = self.model_params['clip']
        self.Wq_loc = nn.Linear(self.embedding_dim * 2, self.head_num * self.qkv_dim, bias=False)
        self.Wq_rout = nn.Linear(self.embedding_dim * 2 + 1, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)
        self.placeholder = nn.Parameter(torch.Tensor(self.embedding_dim))
        self.placeholder.data.uniform_(-1, 1)
        self.placeholder_load = nn.Parameter(torch.Tensor(1))
        self.placeholder_load.data.uniform_(-1, 1)
        self.k = None
        self.v = None
        self.depots_key = None
        self.add_key = None
        self.nodes_key = None
        self.q = None
        self.depot_size = None
        self.customer_size = None
        self.node_size = None

    def set_k_v(self, encoded_nodes, reset_state):
        self.depot_size = reset_state.depot_size
        self.customer_size = reset_state.customer_size
        self.mt_size = reset_state.mt_size
        self.node_size = self.depot_size + self.customer_size

        self.k = multi_head_qkv(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = multi_head_qkv(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape: (batch_size, head_num, node_size, qkv_dim)

        self.nodes_key = encoded_nodes.transpose(1, 2)
        # shape: (batch_size, embedding_dim, node_size)

    def forward(self, encoded_nodes, encoded_nodes_mean, state, mask, last_hh=None):
        encoded_nodes_mean = encoded_nodes_mean.repeat(1, mask.size(1), 1)
        if state.current_node is not None:
            encoded_last_node = get_encoding(encoded_nodes, state)
            input_cat = torch.cat((encoded_last_node, encoded_nodes_mean, state.load[:, :, None]), dim=2)
        else:
            encoded_last_node = self.placeholder[None, None, :].expand_as(encoded_nodes_mean).to(encoded_nodes_mean.device)
            load = self.placeholder_load[None, None, :].expand(encoded_nodes_mean.size(0), encoded_nodes_mean.size(1), 1)
            input_cat = torch.cat((encoded_last_node, encoded_nodes_mean, load.to(encoded_nodes_mean.device)), dim=2)
        
        # (1) Construction of Routing Query
        q_routing = self.Wq_rout(input_cat).reshape(-1, self.embedding_dim)[:, None, :]

        # (2) Subtour Embedding Aggregate
        _, mt_size, max_sub_len = state.current_subtour_nodes.size()
        embed_dim = encoded_nodes.size(2)
        encoded_exp = encoded_nodes.unsqueeze(1).expand(-1, mt_size, -1, -1)
        gather_idx = state.current_subtour_nodes.unsqueeze(-1).expand(-1, -1, -1, embed_dim)
        subtour_embs = torch.gather(encoded_exp, 2, gather_idx)
        
        # Mask 1: Padding Mask
        pos = torch.arange(max_sub_len, device=encoded_nodes.device).reshape(1, 1, -1)
        mask_valid_len = pos < state.current_subtour_len.unsqueeze(-1)
        
        # Mask 2: Customer Mask
        mask_is_customer = state.current_subtour_nodes >= self.depot_size
        
        # Mask Combination
        final_mask = mask_valid_len & mask_is_customer
        final_mask_exp = final_mask.unsqueeze(-1).float()
        
        subtour_sum = (subtour_embs * final_mask_exp).sum(dim=2)
        valid_customer_count = final_mask.sum(dim=2, keepdim=True).float()
        valid_customer_count = torch.clamp(valid_customer_count, min=1.0)
        subtour_mean = subtour_sum / valid_customer_count
        input_cat_loc = torch.cat((encoded_last_node, subtour_mean), dim=2)
        
        # (3) Construction of Location Query
        q_location = self.Wq_loc(input_cat_loc).reshape(-1, self.embedding_dim)[:, None, :]

        if last_hh is None:
            # First Step
            last_hh = encoded_nodes_mean.reshape(-1, self.embedding_dim)[None, :, :].clone()
            self.q = multi_head_qkv(q_routing.reshape(-1, mask.size(1), self.embedding_dim), head_num=self.head_num)
        else:
            last_node = (state.selected_node_list[:, :, -2] < self.depot_size).reshape(1, -1, 1)
            node = (state.current_node < self.depot_size).reshape(1, -1, 1)
            # Indicator State
            flag = (~last_node * node).int()
            flag_reshaped = flag.squeeze(0).unsqueeze(1)
            final_query = q_location * flag_reshaped + q_routing * (1 - flag_reshaped)
        
            self.q = multi_head_qkv(final_query.reshape(-1, mask.size(1), self.embedding_dim), head_num=self.head_num)


        attention_nodes = multi_head_attention(self.q, self.k, self.v, rank3_mask=mask)

        # shape: (batch_size, mt_size, head_num * qkv_dim)
        score = self.multi_head_combine(attention_nodes)
        # shape: (batch_size, mt_size, embedding_dim)
        score_nodes = torch.matmul(score, self.nodes_key)
        # shape: (batch_size, mt_size, node_size)
        sqrt_embedding_dim = self.embedding_dim ** (1 / 2)
        score_scaled = score_nodes / sqrt_embedding_dim
        # shape: (batch_size, mt_size, node_size)
        score_clipped = self.clip * torch.tanh(score_scaled)
        score_masked = score_clipped + mask
        prob = F.softmax(score_masked, dim=2)
        # shape: (batch_size, mt_size, node_size)
        return prob, last_hh


class Norm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(self.embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # shape: (batch_size, node_size, embedding_dim)
        input_added = input1 + input2
        # shape: (batch_size, node_size, embedding_dim)
        input_transposed = input_added.transpose(1, 2)
        # shape: (batch_size, embedding_dim, node_size)
        input_normed = self.norm(input_transposed)
        # shape: (batch_size, embedding_dim, node_size)
        output_transposed = input_normed.transpose(1, 2)
        # shape: (batch_size, node_size, embedding_dim)
        return output_transposed


class FF(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(self.embedding_dim, self.ff_hidden_dim)
        self.W2 = nn.Linear(self.ff_hidden_dim, self.embedding_dim)

    def forward(self, input1):
        # shape: (batch_size, node_size, embedding_dim)
        return self.W2(F.relu(self.W1(input1)))

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        # Initialize layers
        self.layers = nn.ModuleList()
        # Add the first layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # Add hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        # Forward pass through each layer
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))  # Apply ReLU activation function to each hidden layer
        # No activation function for the output layer
        x = self.layers[-1](x)
        return x
    

def get_encoding(encoded_nodes, state):
    # encoded_customers shape: (batch_size, node_size, embedding_dim)
    # index_to_pick shape: (batch_size, mt_size)
    index_to_pick = state.current_node
    batch_size = index_to_pick.size(0)
    mt_size = index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    index_to_gather = index_to_pick[:, :, None].expand(batch_size, mt_size, embedding_dim)
    # shape: (batch_size, mt_size, embedding_dim)
    picked_customers = encoded_nodes.gather(dim=1, index=index_to_gather)
    # shape: (batch_size, mt_size, embedding_dim)
    return picked_customers


def multi_head_qkv(qkv, head_num):
    # shape: (batch_size, n, embedding_dim) : n can be 1 or node_size
    batch_size = qkv.size(0)
    n = qkv.size(1)
    qkv_multi_head = qkv.reshape(batch_size, n, head_num, -1)
    qkv_transposed = qkv_multi_head.transpose(1, 2)
    # shape: (batch_size, head_num, n, key_dim)
    return qkv_transposed


def multi_head_attention(q, k, v, rank2_mask=None, rank3_mask=None):
    # q shape: (batch_size, head_num, n, key_dim)
    # k,v shape: (batch_size, head_num, node_size, key_dim)
    # rank2_mask shape: (batch_size, node_size)
    # rank3_mask shape: (batch_size, group, node_size)
    batch_size = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    depot_customer_size = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape :(batch_size, head_num, n, node_size)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_mask is not None:
        score_scaled = score_scaled + rank2_mask[:, None, None, :].expand(batch_size, head_num, n, depot_customer_size)
    if rank3_mask is not None:
        score_scaled = score_scaled + rank3_mask[:, None, :, :].expand(batch_size, head_num, n, depot_customer_size)
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch_size, head_num, n, node_size)
    out = torch.matmul(weights, v)
    # shape: (batch_size, head_num. n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch_size, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_size, n, head_num * key_dim)
    # shape: (batch_size, n, head_num * key_dim)
    return out_concat

