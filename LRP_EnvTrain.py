from dataclasses import dataclass
import torch
from LRP_Problem import get_random_problems_lrp


@dataclass
class Reset_State:
    depot_x_y: torch.Tensor = None
    # shape: (batch_size, depot_size, 2)
    customer_x_y: torch.Tensor = None
    # shape: (batch_size, customer_size, 2)
    customer_demand: torch.Tensor = None
    # shape: (batch_size, customer_size)
    depot_size: torch.int = None
    mt_size: torch.int = None
    customer_size: torch.int = None
    vehicle_capacity: torch.int = None
    depot_capacity: torch.Tensor = None
    depot_cost: torch.Tensor = None
    vehicle_cost: torch.int = None

    



@dataclass
class Step_State:
    batch_idx: torch.Tensor = None
    mt_idx: torch.Tensor = None
    selected_count: int = None
    current_node: torch.Tensor = None
    finished: torch.Tensor = None
    # shape: (batch_size, mt_size)
    mask: torch.Tensor = None
    # shape: (batch_size, mt_size, node)
    depot_mask: torch.Tensor = None
    # shape: (batch_size, mt_size, depot)
    depot_mask_full: torch.Tensor = None
    selected_node_list: torch.Tensor = None
    load_depot: torch.Tensor = None
    load: torch.Tensor = None
    subtour_count: torch.Tensor = None
    # Subtour Record
    current_subtour_nodes: torch.Tensor = None
    current_subtour_len: torch.Tensor = None

class LRPEnvTrain:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.customer_size = self.env_params['customer_size']
        self.depot_size = self.env_params['depot_size']
        # self.mt_size = self.env_params['mt_size'] * self.depot_size
        self.mt_size = self.env_params['mt_size']
        self.depot_capacity_min = self.env_params['depot_capacity_min']
        self.depot_capacity_max = self.env_params['depot_capacity_max']
        self.depot_cost_min = self.env_params['depot_cost_min']
        self.depot_cost_max = self.env_params['depot_cost_max']
        self.vehicle_capacity = self.env_params['vehicle_capacity']
        self.vehicle_cost = self.env_params['vehicle_cost']
        self.demand_min = self.env_params['demand_min']
        self.demand_max = self.env_params['demand_max']
        self.max_subtour_len = self.customer_size * 2




        self.batch_idx = None
        self.mt_idx = None
        # shape: (batch_size, mt_size)
        self.batch_size = None
        self.depot_customer_x_y = None
        # shape: (batch_size, node, 2)
        self.depot_customer_demand = None
        # shape: (batch_size, node)

        self.selected_count = None
        self.current_node = None
        # shape: (batch_size, mt_size)
        self.selected_node_list = None
        # shape: (batch_size, mt_size, 0~)

        self.departure_depot = None
        self.at_the_depot = None
        self.last_at_the_depot = None
        # shape: (batch_size, mt_size)
        self.load = None
        # shape: (batch_size, mt_size)
        self.load_depot = None
        # shape: (batch_size, mt_size, depot)
        self.visited_flag = None
        # shape: (batch_size, mt_size, node)
        self.visited_flag_backup = None
        # shape: (batch_size, mt_size, node)
        self.mask = None
        # shape: (batch_size, mt_size, node)
        self.finished = None
        # shape: (batch_size, mt_size)
        self.depot_mask = None
        # shape: (batch_size, mt_size, depot)
        self.depot_capacity = None
        self.depot_cost = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, device, aug_factor=1):

        self.batch_size = batch_size
        depot_x_y, depot_capacity, depot_cost, customer_x_y, customer_demand, _ = get_random_problems_lrp(
            batch_size=batch_size, depot_size=self.depot_size, customer_size=self.customer_size, 
            depot_capacity_min=self.depot_capacity_min, depot_capacity_max=self.depot_capacity_max, 
            depot_cost_min=self.depot_cost_min, depot_cost_max=self.depot_cost_max,
            vehicle_capacity=self.vehicle_capacity, vehicle_cost=self.vehicle_cost, 
            demand_min=self.demand_min, demand_max=self.demand_max, aug_type=None)

        depot_x_y = depot_x_y.to(device)
        depot_capacity = depot_capacity.to(device)
        depot_cost = depot_cost.to(device)
        customer_x_y = customer_x_y.to(device)
        customer_demand = customer_demand.to(device)

        self.depot_customer_x_y = torch.cat((depot_x_y, customer_x_y), dim=1)
        # shape: (batch_size, node, 2)
        depot_demand = torch.zeros(size=(self.batch_size, self.depot_size))
        # shape: (batch_size, depot_size)
        self.depot_customer_demand = torch.cat((depot_demand, customer_demand), dim=1)
        # shape: (batch_size, node)
        self.batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.mt_size)
        self.mt_idx = torch.arange(self.mt_size)[None, :].expand(self.batch_size, self.mt_size)
        self.depot_cost = depot_cost

        self.reset_state.depot_x_y = depot_x_y
        self.reset_state.customer_x_y = customer_x_y
        self.reset_state.customer_demand = customer_demand
        self.reset_state.depot_size = self.depot_size
        self.reset_state.customer_size = self.customer_size
        self.reset_state.mt_size = self.mt_size
        self.reset_state.vehicle_capacity = self.vehicle_capacity
        self.reset_state.depot_capacity = depot_capacity
        self.reset_state.depot_cost = depot_cost
        self.reset_state.vehicle_cost = self.vehicle_cost


        self.step_state.batch_idx = self.batch_idx
        self.step_state.mt_idx = self.mt_idx

    def reset(self):
        self.selected_count = 0
        self.subtour_count = torch.zeros((self.batch_size, self.mt_size), dtype=torch.long)
        self.current_node = None
        # shape: (batch_size, mt_size)
        self.selected_node_list = torch.zeros((self.batch_size, self.mt_size, 0), dtype=torch.long)
        # shape: (batch_size, mt_size, 0~)
        self.departure_depot = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.long)
        # shape: (batch_size, mt_size)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch_size, mt_size)
        self.last_at_the_depot = torch.ones(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch_size, mt_size)
        self.load = torch.ones(size=(self.batch_size, self.mt_size))
        # shape: (batch_size, mt_size)
        self.visited_flag = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch_size, mt_size, node)
        self.visited_flag_backup = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch_size, mt_size, node)
        self.mask = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        self.mask[:, :, self.depot_size:] = float('-inf')
        # shape: (batch_size, mt_size, node)
        self.depot_mask = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size))
        # shape: (batch_size, mt_size, depot)
        self.depot_mask_full = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size + self.customer_size))
        # shape: (batch_size, mt_size, depot)
        self.depot_mask_backup = torch.zeros(size=(self.batch_size, self.mt_size, self.depot_size))
        # shape: (batch_size, mt_size, depot)
        self.finished = torch.zeros(size=(self.batch_size, self.mt_size), dtype=torch.bool)
        # shape: (batch_size, mt_size)

        self.cur_subtour_nodes = torch.zeros(
            (self.batch_size, self.mt_size, self.max_subtour_len), 
            dtype=torch.long, 
            device=self.depot_customer_x_y.device
        )
        # shape: (batch_size, mt_size, max_subtour_len)
        self.cur_subtour_len = torch.zeros(
            (self.batch_size, self.mt_size), 
            dtype=torch.long, 
            device=self.depot_customer_x_y.device
        )
        # shape: (batch_size, mt_size)
        self.load_depot = self.reset_state.depot_capacity[:, None, :].expand(self.batch_size, self.mt_size, self.depot_size).clone()
        self.depot_capacity = self.reset_state.depot_capacity[:, None, :].expand(self.batch_size, self.mt_size, self.depot_size).clone()
        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.subtour_count = self.subtour_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.depot_mask = self.depot_mask
        zeros_to_add = torch.zeros(size=(self.batch_size, self.mt_size, self.customer_size))
        self.step_state.depot_mask_full = torch.cat((self.step_state.depot_mask, zeros_to_add), dim=-1)
        self.step_state.finished = self.finished

        self.step_state.current_subtour_nodes = self.cur_subtour_nodes
        self.step_state.current_subtour_len = self.cur_subtour_len

        reward = None
        done = False
        return self.step_state, reward, done


    def step(self, selected, epoch):
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch_size, mt_size)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        done = self.finished.all()        
        # shape: (batch_size, mt_size, 0~)

        # Whether current node is depot
        self.at_the_depot = (selected < self.depot_size)
        # Record the departure depot
        self.departure_depot[self.at_the_depot] = self.current_node[self.at_the_depot]

        # Obtain the demand information (demand of depot is 0)
        demand_list = self.depot_customer_demand[:, None, :].expand(self.batch_size, self.mt_size, -1).clone()
        index_to_gather = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=index_to_gather).squeeze(dim=2)

        # Update vehicle load
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill at the depot
        # Update depot load
        self.load_depot[self.batch_idx, self.mt_idx, self.departure_depot] = self.load_depot[self.batch_idx, self.mt_idx, self.departure_depot].clone() - selected_demand

        # Masking Rule 1
        self.visited_flag[self.batch_idx, self.mt_idx, selected] = float('-inf')
        self.visited_flag[:, :, 0:self.depot_size][self.at_the_depot] = float('-inf')
        self.mask = self.visited_flag.clone()
        
        # Masking Rule 3
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        depot_demand_too_large = self.load_depot[self.batch_idx, self.mt_idx, self.departure_depot][:, :, None] + round_error_epsilon < demand_list
        self.mask[demand_too_large] = float('-inf')
        self.mask[depot_demand_too_large] = float('-inf')

        # Masking Rule 4
        condition_1 = self.last_at_the_depot & self.at_the_depot
        self.mask[:, :, 0:self.depot_size][condition_1]=float('-inf')


        # Subtour Information Update
        self.cur_subtour_nodes = self.cur_subtour_nodes.clone()
        self.cur_subtour_len = self.cur_subtour_len.clone()
        self.load = self.load.clone()
        self.load_depot = self.load_depot.clone()
        
        if condition_1.any():
            self.cur_subtour_nodes[condition_1] = 0
            self.cur_subtour_len[condition_1] = 0
            
        scatter_index = self.cur_subtour_len[:, :, None]
        self.cur_subtour_nodes.scatter_(2, scatter_index, selected[:, :, None])
        self.cur_subtour_len += 1

        # Masking Rule 5
        condition_2 = (~self.last_at_the_depot) & self.at_the_depot
        self.subtour_count[condition_2] = self.subtour_count[condition_2] + 1
        if condition_2.any():
            inf_tensor = torch.full_like(self.depot_mask, float('-inf'), device=self.load.device)
            adjusted_demand = demand_list.clone()
            idx_visited = self.visited_flag==float('-inf')
            adjusted_demand[idx_visited]=float('inf')
            values, _ = torch.min(adjusted_demand.reshape(-1,self.customer_size + self.depot_size), dim=1)
            values = values.reshape(self.batch_size, self.mt_size)[:,:,None].expand(self.batch_size, self.mt_size, self.depot_size)
            adequate_depot = self.load_depot >= values

            self.mask[:, :, 0:self.depot_size][condition_2] = torch.where(
                adequate_depot[condition_2],
                self.depot_mask[condition_2],
                inf_tensor[condition_2]
            )
            self.mask[:, :, self.depot_size:][condition_2] = float('-inf')
            




        # Masking Rule 2
        condition_3 = ((~self.last_at_the_depot) & (~self.at_the_depot)) | ((self.last_at_the_depot) & (~self.at_the_depot))
        self.mask[:, :, 0:self.depot_size][condition_3] = float('-inf')

        _expanded_condition_3 = torch.zeros((self.batch_size, self.mt_size, self.depot_size), dtype=torch.bool)
        expanded_condition_3 = _expanded_condition_3.scatter(2, self.departure_depot.unsqueeze(2), condition_3.unsqueeze(2))
        self.mask[:, :, 0:self.depot_size] = torch.where(expanded_condition_3, torch.zeros_like(self.mask[:, :, 0:self.depot_size]), self.mask[:, :, 0:self.depot_size])

        if done:
            reward = -self.get_travel_distance_lrp(epoch=epoch)
            self.step_state.result_node_list = self.selected_node_list
        else:
            reward = None

        self.last_at_the_depot = self.at_the_depot

        new_finished = (self.visited_flag[:, :, self.depot_size:] == float('-inf')).all(dim=2)
        self.mask[:, :, 0:self.depot_size][self.finished] = 0  # do not mask depot for finished episode.
        self.finished = self.finished + new_finished
        

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.mask = self.mask
        self.step_state.finished = self.finished
        self.step_state.selected_node_list = self.selected_node_list
        self.step_state.load_depot = self.load_depot
        self.step_state.load = self.load
        self.step_state.subtour_count = self.subtour_count

        self.step_state.current_subtour_nodes = self.cur_subtour_nodes
        self.step_state.current_subtour_len = self.cur_subtour_len

        

        return self.step_state, reward, done

    def get_travel_distance(self):
        # Travel Distances
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_x_y = self.depot_customer_x_y[:, None, :, :].expand(-1, self.mt_size, -1, -1)
        seq_ordered = all_x_y.gather(dim=2, index=index_to_gather)
        depot_ordered = self.selected_node_list < self.depot_size
        depot_rolled = depot_ordered.roll(dims=2, shifts=-1)
        depot_final = depot_ordered * depot_rolled
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        segment_lengths[depot_final] = 0
        travel_distances = segment_lengths.sum(2)

        return travel_distances

    def get_travel_distance_lrp(self, epoch):
        # Travel Distances
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_x_y = self.depot_customer_x_y[:, None, :, :].expand(-1, self.mt_size, -1, -1)
        seq_ordered = all_x_y.gather(dim=2, index=index_to_gather)
        depot_ordered = self.selected_node_list < self.depot_size
        depot_rolled = depot_ordered.roll(dims=2, shifts=-1)
        depot_final = depot_ordered * depot_rolled
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        segment_lengths[depot_final] = 0
        travel_distances = segment_lengths.sum(2)

        # Vehicle Cost
        final_vehicle_cost = self.subtour_count * self.vehicle_cost

        # Depot Cost
        tmp = self.depot_capacity - self.load_depot
        opened = tmp != 0
        unopened = tmp == 0
        tmp[opened] = 1
        tmp[unopened] = 0
        depot_cost_each = tmp * self.depot_cost[:,None,:]
        final_depot_cost = torch.sum(depot_cost_each,-1)

        total_cost = travel_distances + final_vehicle_cost + final_depot_cost

        return total_cost
