'''
Date: 2024-03-11 18:00:06
LastEditors: JackeyMiao
LastEditTime: 2026-03-23 14:46:25
FilePath: /1_DRLHQ/data/gen_data.py
'''
import argparse
import os
import numpy as np
import torch
import pickle

# from plot import display_points_with_pmedian

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

# from utils.data_utils import check_extension, save_dataset


def generate_LRP_data(n_samples, depot_size, customer_size, 
                    depot_capacity_min=500, depot_capacity_max=800, depot_cost_min=100, depot_cost_max=200, 
                    vehicle_capacity=50, vehicle_cost=20, demand_min=1, demand_max=10):
    data = []
    node_size = depot_size + customer_size
    for _ in range(n_samples):
        depot_x_y = torch.rand(size=(depot_size, 2))
        # shape: (batch, depot_size, 2)
        # depot_capacity = torch.randint(depot_capacity_min, depot_capacity_max, size=(depot_size, )) #/ vehicle_capacity
        # depot_cost = torch.randint(depot_cost_min, depot_cost_max, size=(depot_size, ))
        depot_capacity = (torch.rand(depot_size) * (depot_capacity_max - depot_capacity_min) + depot_capacity_min) # / vehicle_capacity
        depot_cost = torch.rand(depot_size) * (depot_cost_max - depot_cost_min) + depot_cost_min

        customer_x_y = torch.rand(size=(customer_size, 2))
        # shape: (batch, customer_size, 2)
        customer_demand = torch.randint(demand_min, demand_max, size=(customer_size, )) #/ vehicle_capacity
        data.append(dict(depot_x_y=depot_x_y,
                    depot_capacity=depot_capacity,
                    depot_cost=depot_cost,
                    customer_x_y=customer_x_y,
                    customer_demand=customer_demand,
                    vehicle_cost=vehicle_cost,
                    vehicle_capacity=vehicle_capacity,
                    name=_,
                    # n_samples=n_samples,
                    ))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset")
    parser.add_argument('--depot_size', type=int, default=40,
                        help="number of depots")
    parser.add_argument('--customer_size', type=int, default=600,
                        help="number of customers")
    parser.add_argument('--depot_capacity_min', type=int, default=500,
                        help="min value of depot capacity")
    parser.add_argument('--depot_capacity_max', type=int, default=900,
                        help="max value of depot capacity")
    parser.add_argument('--depot_cost_min', type=int, default=5,
                        help="min value of depot cost")
    parser.add_argument('--depot_cost_max', type=int, default=15,
                        help="max value of depot cost")
    parser.add_argument('--vehicle_capacity', type=int, default=70,
                        help="capacity of vehicle")
    parser.add_argument('--vehicle_cost', type=int, default=1,
                        help="cost of vehicle")
    parser.add_argument('--demand_min', type=int, default=10,
                        help="min value of customer demand")
    parser.add_argument('--demand_max', type=int, default=20,
                        help="max value of customer demand")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    

    opts = parser.parse_args()

    assert opts.filename is None, \
        "Can only specify filename when generating a single dataset"

    torch.manual_seed(1234)

    datadir = opts.data_dir
    os.makedirs(datadir, exist_ok=True)
    
    depot_size = opts.depot_size
    customer_size = opts.customer_size
    n_samples = opts.dataset_size


    filename = os.path.join(datadir, f"{depot_size}_{customer_size}_{n_samples}.pkl")

    dataset = generate_LRP_data(n_samples, depot_size, customer_size, 
                                opts.depot_capacity_min, opts.depot_capacity_max, opts.depot_cost_min, opts.depot_cost_max, 
                                opts.vehicle_capacity, opts.vehicle_cost, opts.demand_min, opts.demand_max)
    
    save_dataset(dataset, filename)



