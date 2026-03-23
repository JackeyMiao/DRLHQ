import torch
import numpy as np
import pandas as pd
import copy


def get_random_problems(batch_size, depot_size, customer_size, capacity=50, demand_min=1, demand_max=10,
                        aug_type=None):
    node_size = depot_size + customer_size
    
    depot_x_y = torch.rand(size=(batch_size, depot_size, 2))
    # shape: (batch, depot_size, 2)
    customer_x_y = torch.rand(size=(batch_size, customer_size, 2))
    # shape: (batch, customer_size, 2)
    customer_demand = torch.randint(demand_min, demand_max, size=(batch_size, customer_size)) / capacity
    depot_x_y, customer_x_y, customer_demand, aug_factor = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand)

    return depot_x_y, customer_x_y, customer_demand, aug_factor

def get_random_problems_lrp(batch_size, depot_size, customer_size, 
                            depot_capacity_min=500, depot_capacity_max=800, depot_cost_min=100, depot_cost_max=200, 
                            vehicle_capacity=50, vehicle_cost=20, demand_min=1, demand_max=10, 
                            aug_type=None):
    node_size = depot_size + customer_size

    depot_x_y = torch.rand(size=(batch_size, depot_size, 2))
    customer_x_y = torch.rand(size=(batch_size, customer_size, 2))

    depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=1)

    # Calculate the maximum and minimum values along each coordinate axis across all nodes, per batch
    max_values, _ = depot_customer_x_y.max(dim=1, keepdim=True)
    min_values, _ = depot_customer_x_y.min(dim=1, keepdim=True)

    # Max and min for normalizing (separately for x and y coordinates)
    x_max = max_values[:, :, 0]
    y_max = max_values[:, :, 1]
    x_min = min_values[:, :, 0]
    y_min = min_values[:, :, 1]

    # Create a tensor that can be used to normalize all coordinates in the batch
    tmp = torch.stack([x_min, y_min], dim=-1).expand_as(depot_customer_x_y)
    factor = torch.max(x_max - x_min, y_max - y_min).unsqueeze(-1)

    # Normalize coordinates
    depot_customer_x_y_norm = (depot_customer_x_y - tmp) / factor

    depot_x_y = depot_customer_x_y_norm[:, :depot_size, :]
    customer_x_y = depot_customer_x_y_norm[:, depot_size:, :]


    # shape: (batch, depot_size, 2)
    depot_capacity = (torch.rand(batch_size, depot_size) * (depot_capacity_max - depot_capacity_min) + depot_capacity_min) / vehicle_capacity
    depot_cost = torch.rand(batch_size, depot_size) * (depot_cost_max - depot_cost_min) + depot_cost_min

    
    # shape: (batch, customer_size, 2)
    customer_demand = torch.randint(demand_min, demand_max, size=(batch_size, customer_size)) / vehicle_capacity

    depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand, depot_capacity=depot_capacity, depot_cost=depot_cost)

    return depot_x_y, depot_capacity, depot_cost, customer_x_y, customer_demand, aug_factor

def get_1_dataset_lrp(batch_size, instance, aug_type):
    depot_size = instance['depot_x_y'].size()[0]

    vehicle_capacity = instance['vehicle_capacity']
    depot_x_y = instance['depot_x_y'].expand(batch_size,-1,-1)
    depot_capacity = instance['depot_capacity'].expand(batch_size,-1) / vehicle_capacity
    depot_cost = instance['depot_cost'].expand(batch_size,-1)
    customer_x_y = instance['customer_x_y'].expand(batch_size,-1,-1)
    customer_demand = instance['customer_demand'].expand(batch_size,-1) / vehicle_capacity
    vehicle_cost = instance['vehicle_cost'] 

    # Augmentation
    depot_size = depot_x_y.size()[1]
    customer_size = customer_x_y.size()[1]
    depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost = aug_rotation(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand, depot_capacity=depot_capacity, depot_cost=depot_cost)
    
    depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=1)

    # Calculate the maximum and minimum values along each coordinate axis across all nodes, per batch
    max_values, _ = depot_customer_x_y.max(dim=1, keepdim=True)
    min_values, _ = depot_customer_x_y.min(dim=1, keepdim=True)

    # Max and min for normalizing (separately for x and y coordinates)
    x_max = max_values[:, :, 0]
    y_max = max_values[:, :, 1]
    x_min = min_values[:, :, 0]
    y_min = min_values[:, :, 1]

    # Create a tensor that can be used to normalize all coordinates in the batch
    tmp = torch.stack([x_min, y_min], dim=-1).expand_as(depot_customer_x_y)
    factor = torch.max(x_max - x_min, y_max - y_min).unsqueeze(-1)

    # Normalize coordinates
    depot_customer_x_y_norm = (depot_customer_x_y - tmp) / factor
    # depot_customer_x_y_norm = depot_customer_x_y

    depot_x_y = depot_customer_x_y_norm[:, :depot_size, :]
    customer_x_y = depot_customer_x_y_norm[:, depot_size:, :]


    return depot_x_y, depot_capacity, depot_cost, customer_x_y, customer_demand, aug_factor, vehicle_cost, vehicle_capacity, depot_size, customer_size, depot_customer_x_y


def get_batch_dataset_lrp(batch_size, instance, aug_type):
    dict= batch_concatenate_dicts(instance)

    depot_size = dict['depot_x_y'].size()[1]

    vehicle_capacity = dict['vehicle_capacity']
    depot_x_y = dict['depot_x_y']
    depot_capacity = dict['depot_capacity'] / vehicle_capacity
    depot_cost = dict['depot_cost']
    customer_x_y = dict['customer_x_y']
    customer_demand = dict['customer_demand'] / vehicle_capacity
    vehicle_cost = dict['vehicle_cost'] 

    # Augmentation
    depot_size = depot_x_y.size()[1]
    customer_size = customer_x_y.size()[1]
    depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost = aug_rotation(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand, depot_capacity=depot_capacity, depot_cost=depot_cost)
    
    depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=1)

    # Calculate the maximum and minimum values along each coordinate axis across all nodes, per batch
    max_values, _ = depot_customer_x_y.max(dim=1, keepdim=True)
    min_values, _ = depot_customer_x_y.min(dim=1, keepdim=True)

    # Max and min for normalizing (separately for x and y coordinates)
    x_max = max_values[:, :, 0]
    y_max = max_values[:, :, 1]
    x_min = min_values[:, :, 0]
    y_min = min_values[:, :, 1]

    # Create a tensor that can be used to normalize all coordinates in the batch
    tmp = torch.stack([x_min, y_min], dim=-1).expand_as(depot_customer_x_y)
    factor = torch.max(x_max - x_min, y_max - y_min).unsqueeze(-1)

    # Normalize coordinates
    depot_customer_x_y_norm = (depot_customer_x_y - tmp) / factor
    

    depot_x_y = depot_customer_x_y_norm[:, :depot_size, :]
    customer_x_y = depot_customer_x_y_norm[:, depot_size:, :]


    return depot_x_y, depot_capacity, depot_cost, customer_x_y, customer_demand, aug_factor, vehicle_cost, vehicle_capacity, depot_size, customer_size, depot_customer_x_y




def get_dataset_problem(load_path, batch_size, aug_type='d'):

    filename = load_path
    data = pd.read_csv(filename, sep=',', header=None)
    data = data.to_numpy()
    depot_size = int(data[0][0])
    customer_size = int(data[0][1])
    capacity = int(data[0][2])
    scale = int(data[0][3])
    depot_xyd = data[1:depot_size + 1]
    customer_xyd = data[depot_size + 1:depot_size + customer_size + 1]
    full_node = data[1:depot_size + customer_size + 1]
    for i in range(len(depot_xyd)):
        depot_x_y = torch.FloatTensor(depot_xyd[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [depot_x_y, torch.FloatTensor(depot_xyd[i][0:2]).unsqueeze(0)], dim=0)
    for i in range(len(customer_xyd)):
        customer_x_y = torch.FloatTensor(customer_xyd[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
            [customer_x_y, torch.FloatTensor(customer_xyd[i][0:2]).unsqueeze(0)], dim=0)
        customer_demand = torch.FloatTensor(customer_xyd[i][2:3]) if i == 0 else torch.cat(
            [customer_demand, torch.FloatTensor(customer_xyd[i][2:3])], dim=0)
    customer_demand = customer_demand / capacity
    depot_x_y = depot_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    customer_x_y = customer_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
    customer_demand = customer_demand.unsqueeze(0).repeat(batch_size, 1)
    depot_x_y, customer_x_y, customer_demand, aug_factor = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand)
    data = {'depot_x_y': depot_x_y.numpy().tolist(), 'customer_x_y': customer_x_y.numpy().tolist(),
            'customer_demand': customer_demand.numpy().tolist(), 'capacity': capacity, 'full_node': full_node,
            'scale': scale, 'aug_factor': aug_factor}
    return depot_x_y, customer_x_y, customer_demand, depot_size, customer_size, capacity, data, aug_factor


def get_1_random_problems(batch_size, depot_size, customer_size, capacity=50, demand_min=1, demand_max=10,
                          aug_type=None):
    depot_x_y = torch.rand(size=(depot_size, 2)).repeat(batch_size, 1, 1)

    customer_x_y = torch.rand(size=(customer_size, 2)).repeat(batch_size, 1, 1)

    customer_demand = torch.randint(demand_min, demand_max, size=(1, customer_size)).repeat(batch_size, 1) / capacity

    depot_x_y, customer_x_y, customer_demand, aug_factor = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand)
    return depot_x_y, customer_x_y, customer_demand, aug_factor

def get_1_random_problems_lrp(batch_size, depot_size, customer_size, 
                            depot_capacity_min=500, depot_capacity_max=800, depot_cost_min=100, depot_cost_max=200, 
                            vehicle_capacity=50, vehicle_cost=20, demand_min=1, demand_max=10, 
                            aug_type=None):
    node_size = depot_size + customer_size

    depot_x_y = torch.rand(size=(depot_size, 2))
    customer_x_y = torch.rand(size=(customer_size, 2))

    depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=0)
    # Normalization
    max_values, _ = (depot_customer_x_y).max(dim=0)
    min_values, _ = (depot_customer_x_y).min(dim=0)
    x_max = max_values[0]
    y_max = max_values[1]
    x_min = min_values[0]
    y_min = min_values[1]
    tmp = torch.FloatTensor([x_min, y_min]).expand_as(depot_customer_x_y).to(depot_customer_x_y.device)
    factor = max(x_max - x_min, y_max - y_min)
    depot_customer_x_y_norm = (depot_customer_x_y - tmp) / factor

    depot_x_y = depot_customer_x_y_norm[:depot_size,:].expand(batch_size,-1,-1)
    customer_x_y = depot_customer_x_y_norm[depot_size:,:].expand(batch_size,-1,-1)

    # shape: (batch, depot_size, 2)
    depot_capacity = torch.randint(depot_capacity_min, depot_capacity_max, size=(1, depot_size)).repeat(batch_size, 1) / vehicle_capacity
    depot_cost = torch.randint(depot_cost_min, depot_cost_max, size=(1, depot_size)).repeat(batch_size, 1)

    
    # shape: (batch, customer_size, 2)
    customer_demand = torch.randint(demand_min, demand_max, size=(1, customer_size)).repeat(batch_size, 1) / vehicle_capacity

    depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand, depot_capacity=depot_capacity, depot_cost=depot_cost)

    return depot_x_y, depot_capacity, depot_cost, customer_x_y, customer_demand, aug_factor, depot_customer_x_y

def get_1_syn_dataset_lrp(batch_size, instance, aug_type):
    depot_size=instance['depot_x_y'].size()[0]
    
    depot_customer_x_y = torch.cat((instance['depot_x_y'], instance['customer_x_y']), dim=0)
    # Normalization
    max_values, _ = (depot_customer_x_y).max(dim=0)
    min_values, _ = (depot_customer_x_y).min(dim=0)
    x_max = max_values[0]
    y_max = max_values[1]
    x_min = min_values[0]
    y_min = min_values[1]
    tmp = torch.FloatTensor([x_min, y_min]).expand_as(depot_customer_x_y)
    factor = max(x_max - x_min, y_max - y_min)
    depot_customer_x_y_norm = (depot_customer_x_y - tmp) / factor

    vehicle_capacity = instance['vehicle_capacity']
    depot_x_y = depot_customer_x_y_norm[:depot_size,:].expand(batch_size,-1,-1)
    depot_capacity = instance['depot_capacity'].expand(batch_size,-1) / vehicle_capacity
    depot_cost = instance['depot_cost'].expand(batch_size,-1)
    customer_x_y = depot_customer_x_y_norm[depot_size:,:].expand(batch_size,-1,-1)
    customer_demand = instance['customer_demand'].expand(batch_size,-1) / vehicle_capacity
    vehicle_cost = instance['vehicle_cost'] 

    depot_size = depot_x_y.size()[1]
    customer_size = customer_x_y.size()[1]
    depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost = aug(
        aug_type=aug_type, depot_size=depot_size, depot_x_y=depot_x_y, customer_x_y=customer_x_y,
        customer_demand=customer_demand, depot_capacity=depot_capacity, depot_cost=depot_cost)

    return depot_x_y, depot_capacity, depot_cost, customer_x_y, customer_demand, aug_factor, vehicle_cost, vehicle_capacity, depot_size, customer_size, depot_customer_x_y


def aug_rotation(aug_type, depot_size, depot_x_y, customer_x_y, customer_demand, depot_capacity, depot_cost):
    if aug_type is not None:
        n_augmentations = int(aug_type)
        if n_augmentations <= 0:
            raise ValueError("aug_type must be a positive integer.")
        
        aug_list = torch.Tensor([i * 360 / n_augmentations for i in range(n_augmentations)])
        aug_factor = n_augmentations
    else:
        aug_factor = 1
        return depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost

    aug_rad = torch.deg2rad(aug_list)

    depot_coords = None
    customer_coords = None

    for rad in aug_rad:
        cos_theta, sin_theta = torch.cos(rad), torch.sin(rad)
        rotation_matrix = torch.Tensor([[cos_theta, -sin_theta],
                                        [sin_theta, cos_theta]])
        if depot_coords is None:
            depot_coords = torch.matmul(depot_x_y,rotation_matrix)
            customer_coords = torch.matmul(customer_x_y,rotation_matrix)
        else:
            depot_coords = torch.cat((depot_coords, torch.matmul(depot_x_y,rotation_matrix)), dim = 0)
            customer_coords = torch.cat((customer_coords, torch.matmul(customer_x_y,rotation_matrix)), dim = 0)

    
    customer_demand = customer_demand.repeat(aug_factor, 1)
    depot_capacity = depot_capacity.repeat(aug_factor, 1)
    depot_cost = depot_cost.repeat(aug_factor, 1)

    return depot_coords, customer_coords, customer_demand, aug_factor, depot_capacity, depot_cost


def aug(aug_type, depot_size, depot_x_y, customer_x_y, customer_demand, depot_capacity, depot_cost):
    aug_factor = 1
    if aug_type == 'd':
        aug_factor = depot_size + 7
        depot_x_y = augment_x_y_by_d(depot_x_y, depot_size)
        customer_x_y = augment_x_y_by_d(customer_x_y, depot_size)

    elif aug_type == '8':
        aug_factor = 8
        depot_x_y = augment_x_y_by_8(depot_x_y)
        customer_x_y = augment_x_y_by_8(customer_x_y)
    customer_demand = customer_demand.repeat(aug_factor, 1)
    depot_capacity = depot_capacity.repeat(aug_factor, 1)
    depot_cost = depot_cost.repeat(aug_factor, 1)

    return depot_x_y, customer_x_y, customer_demand, aug_factor, depot_capacity, depot_cost

def augment_x_y_by_8(x_y):
    # shape: (batch, N, 2)

    x = x_y[:, :, [0]]
    y = x_y[:, :, [1]]
    # shape: (batch, N, 1)

    data1 = torch.cat((x, y), dim=2)
    data2 = torch.cat((1 - x, y), dim=2)
    data3 = torch.cat((x, 1 - y), dim=2)
    data4 = torch.cat((1 - x, 1 - y), dim=2)
    data5 = torch.cat((y, x), dim=2)
    data6 = torch.cat((1 - y, x), dim=2)
    data7 = torch.cat((y, 1 - x), dim=2)
    data8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_x_y = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
    # shape: (8 * batch_size, N, 2)
    return aug_x_y


def augment_x_y_by_d(x_y, depot_size):
    # shape: (batch, N, 2)

    x = x_y[:, :, [0]]
    y = x_y[:, :, [1]]
    # shape: (batch, N, 1)

    data1 = torch.cat((x, y), dim=2)
    data2 = torch.cat((1 - x, y), dim=2)
    data3 = torch.cat((x, 1 - y), dim=2)
    data4 = torch.cat((1 - x, 1 - y), dim=2)
    data5 = torch.cat((y, x), dim=2)
    data6 = torch.cat((1 - y, x), dim=2)
    data7 = torch.cat((y, 1 - x), dim=2)
    data8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_x_y = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
    # shape: (8 * batch_size, N, 2)
    for i in range(depot_size-1):
        x_y_temp = copy.deepcopy(x_y).clone()
        x_y_temp[:, 0] = x_y_temp[:, i+1].clone()

        x_y_temp[:, i+1] = x_y[:, 0]
        aug_x_y = torch.cat((aug_x_y, x_y_temp), dim=0)
    return aug_x_y


def batch_concatenate_dicts(dict_list):
    # 初始化一个新的字典
    concatenated_dict = {}
    concatenated_dict['depot_x_y'] = dict_list[0]['depot_x_y'][None, :, :]
    concatenated_dict['depot_capacity'] = dict_list[0]['depot_capacity'][None, :]
    concatenated_dict['depot_cost'] = dict_list[0]['depot_cost'][None, :]
    concatenated_dict['customer_x_y'] = dict_list[0]['customer_x_y'][None, :, :]
    concatenated_dict['customer_demand'] = dict_list[0]['customer_demand'][None, :]
    concatenated_dict['vehicle_cost'] = dict_list[0]['vehicle_cost']
    concatenated_dict['vehicle_capacity'] = dict_list[0]['vehicle_capacity']




    for i in range(1, len(dict_list)):
        concatenated_dict
        d = dict_list[i]
        concatenated_dict['depot_x_y'] = torch.cat((concatenated_dict['depot_x_y'], d['depot_x_y'][None, :, :]))
        concatenated_dict['depot_capacity'] = torch.cat((concatenated_dict['depot_capacity'], d['depot_capacity'][None, :]))
        concatenated_dict['depot_cost'] = torch.cat((concatenated_dict['depot_cost'], d['depot_cost'][None, :]))
        concatenated_dict['customer_x_y'] = torch.cat((concatenated_dict['customer_x_y'], d['customer_x_y'][None, :, :]))
        concatenated_dict['customer_demand'] = torch.cat((concatenated_dict['customer_demand'], d['customer_demand'][None, :]))
    
    return concatenated_dict