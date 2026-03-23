import torch
from logging import getLogger
from LRP_EnvEval import LRPEnvEval
from LRP_Model import LRPModel
from my_utils import *
from LRP_Draw_1_problem import Draw_1_Problem
import copy

class LRPEvaluator:
    def __init__(self,
                 env_params,
                 model_params,
                 eval_params):

        # params
        self.env_params = env_params
        self.model_params = model_params
        self.eval_params = eval_params
        self.episodes = self.eval_params['episodes']
        self.sample_size = self.eval_params['eval_batch_size']
        self.augmentation = self.eval_params['augmentation']

        # result, log
        self.logger = getLogger(name='evaluator')
        self.result_folder = get_result_folder()

        # cuda
        use_cuda = self.eval_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.eval_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)
        else:
            device = torch.device('cpu')
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)
        self.device = device

        # main components
        self.env = LRPEnvEval(**self.env_params)
        self.model = LRPModel(**self.model_params)

        # restore
        self.model_load = self.eval_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**self.model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()


    def run(self):
        self.time_estimator.reset()

        score = AverageMeter()
        score_aug = AverageMeter()
        score_depot_aug_avg = AverageMeter()
        score_dist_aug_avg = AverageMeter()
        score_vehicle_aug_avg = AverageMeter()
        nb_depots_aug_avg = AverageMeter()
        nb_vehicles_aug_avg = AverageMeter()
        episode = 0


        # set test method
        test_method_list = {
            'origin':   self._test_one_batch,
            'bs':     self._test_one_batch_bs, 
        }

        test_method = test_method_list[self.eval_params['mode']]

        filepath = self.env_params['load_path']
        dataset = load_dataset(filepath)
        self.episodes = len(dataset)
        while episode < self.episodes:
            remain_episode = self.episodes - episode

            batch_size = min(self.sample_size, remain_episode)
            self.sample_size = batch_size
            batch = dataset[episode:episode+batch_size]

            if self.sample_size == 1:
                score_avg, score_aug_avg, solution_no_aug, solution_aug, score_dist_aug, score_vehicle_aug, score_depot_aug, nb_depots_aug, nb_vehicles_aug = test_method(batch)
                score.update(score_avg, batch_size)
                score_aug.update(score_aug_avg, batch_size)
                score_depot_aug_avg.update(score_depot_aug, batch_size)
                score_dist_aug_avg.update(score_dist_aug, batch_size)
                score_vehicle_aug_avg.update(score_vehicle_aug, batch_size)
                nb_depots_aug_avg.update(nb_depots_aug, batch_size)
                nb_vehicles_aug_avg.update(nb_vehicles_aug, batch_size)
                episode += batch_size
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, self.episodes)
                instance = batch[0]
                draw_data = copy.deepcopy(instance)
                draw_data['depot_x_y'] = copy.deepcopy(instance['depot_x_y'])
                draw_data['customer_x_y'] = copy.deepcopy(instance['customer_x_y'])
                draw_data['full_node'] = np.vstack((draw_data['depot_x_y'], draw_data['customer_x_y']))
                is_depot = np.zeros([len(draw_data['full_node']), 2])
                for i in range(len(draw_data['depot_x_y'])):
                    is_depot[i] = [1, 1]
                draw_data['full_node'] = np.hstack((draw_data['full_node'], is_depot))
                
                Draw_1_Problem(draw_data, solution_no_aug, self.result_folder, episode, aug=False)
                Draw_1_Problem(draw_data, solution_aug, self.result_folder, episode, aug=True)
                self.logger.info('episode: {:3d}/{:3d}, Elapsed: [{}], Remain: [{}], score: {:.4f}, score_aug: {:.4f}, Name: {}, score_depot_aug: {:.4f}, score_dist_aug: {:.4f}, score_vehicle_aug: {:.4f}, nb_depots_aug: {:.4f}, nb_vehicles_aug: {:.4f}'
                            .format(episode, self.episodes, elapsed_time_str, remain_time_str, score_avg,
                                    score_aug_avg, instance['name'], score_depot_aug, score_dist_aug, score_vehicle_aug, nb_depots_aug, nb_vehicles_aug))
            else:
                score_avg, score_aug_avg = test_method(batch)
                score.update(score_avg, batch_size)
                score_aug.update(score_aug_avg, batch_size)
                episode += batch_size
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, self.episodes)
                self.logger.info('episode: {:3d}/{:3d}, Elapsed: [{}], Remain: [{}], score: {:.4f}, score_aug: {:.4f}'
                            .format(episode, self.episodes, elapsed_time_str, remain_time_str, score_avg,
                                    score_aug_avg))
            
            all_done = (episode == self.episodes)
            if all_done:
                if self.sample_size == 1:
                    self.logger.info('Evaluate done, score without aug: {:.4f}, score with aug: {:.4f}, score_depot_aug: {:.4f}, score_dist_aug: {:.4f}, score_vehicle_aug: {:.4f}, nb_depots_aug: {:.4f}, nb_vehicles_aug: {:.4f}'
                                    .format(score.avg, score_aug.avg, score_depot_aug_avg.avg, score_dist_aug_avg.avg, score_vehicle_aug_avg.avg, nb_depots_aug_avg.avg, nb_vehicles_aug_avg.avg))
                else:
                    self.logger.info('Evaluate done, score without aug: {:.4f}, score with aug: {:.4f}'
                                .format(score.avg, score_aug.avg))
                    
    def _get_pomo_starting_points(self, model, env, num_starting_points):
        # Ready
        ###############################################
        model.eval()
        self.env.modify_pomo_size(self.env_params['sample_size'])
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        last_hh = None
        while not done:
            selected, _, last_hh = self.model(state, last_hh)
            # shape: (batch_size, mt_size)
            state, reward, reward_distances, reward_vehicles, reward_depots, done, nb_depot, nb_vehicle = self.env.step(selected)
        
        # starting points
        ###############################################
        sorted_index = reward.sort(dim=1, descending=True).indices
        selected_index = sorted_index[:, :num_starting_points]
        selected_index = selected_index + self.env.depot_size     # depot is 0-depot_size, and node index starts from depot_size + 1
        # shape: (batch, num_starting_points)
        
        return selected_index   
    
    def _eval(self, instance = None):
        # aug
        aug_type = self.augmentation

        # prepare
        self.model.eval()

        with torch.no_grad():
            self.env.load_batch_problems(self.sample_size, device=self.device, instance=instance, aug_type=aug_type)

            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
        
        # mt rollout
        state, reward, done = self.env.pre_step()
        last_hh = None
        while not done:
            selected, _, last_hh = self.model(state, last_hh)
            # shape: (batch_size, mt_size)
            state, reward, reward_distances, reward_vehicles, reward_depots, done, nb_depot, nb_vehicle = self.env.step(selected)

        reward_batch = torch.hstack(torch.split(reward, self.sample_size))
        solution = torch.hstack(torch.split(state.selected_node_list, self.sample_size))
        
        
        reward_max, reward_max_idx = reward_batch[:, 0:self.env.depot_size].max(1)
        reward_no_aug = reward_max.mean()
        reward_max_aug, reward_max_idx_aug = reward_batch.max(1)
        reward_aug = reward_max_aug.mean()

        score_no_aug = -reward_no_aug.float()
        score_aug = -reward_aug.float()

        if self.sample_size == 1:
            reward_distances_batch = torch.hstack(torch.split(reward_distances, self.sample_size))
            reward_vehicles_batch = torch.hstack(torch.split(reward_vehicles, self.sample_size))
            reward_depots_batch = torch.hstack(torch.split(reward_depots, self.sample_size))
            nb_depots_batch = torch.hstack(torch.split(nb_depot, self.sample_size))
            nb_vehicles_batch = torch.hstack(torch.split(nb_vehicle, self.sample_size))
            solution_no_aug = solution[0][reward_max_idx][0]
            solution_aug = solution[0][reward_max_idx_aug][0]
            score_dist_aug = -reward_distances_batch[0][reward_max_idx_aug].float()
            score_vehicle_aug = -reward_vehicles_batch[0][reward_max_idx_aug].float()
            score_depot_aug = -reward_depots_batch[0][reward_max_idx_aug].float()
            nb_depots_aug = nb_depots_batch[0][reward_max_idx_aug].float()
            nb_vehicles_aug = nb_vehicles_batch[0][reward_max_idx_aug].float()

            return score_no_aug.item(), score_aug.item(), solution_no_aug, solution_aug, score_dist_aug.item(), score_vehicle_aug.item(), score_depot_aug.item(), nb_depots_aug.item(), nb_vehicles_aug.item()

        
        return score_no_aug.item(), score_aug.item()

    def _test_one_batch(self, instance = None):
        # aug
        aug_type = self.augmentation

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_batch_problems(self.sample_size, device=self.device, instance=instance, aug_type=aug_type)

            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        last_hh = None
        while not done:
            selected, _, last_hh = self.model(state, last_hh)
            # shape: (batch_size, mt_size)
            state, reward, reward_distances, reward_vehicles, reward_depots, done, nb_depot, nb_vehicle = self.env.step(selected)
        
        # Return
        ###############################################
        reward_batch = torch.hstack(torch.split(reward, self.sample_size))
        solution = torch.hstack(torch.split(state.selected_node_list, self.sample_size))
        
        
        reward_max, reward_max_idx = reward_batch[:, 0:self.env.depot_size].max(1)
        reward_no_aug = reward_max.mean()
        reward_max_aug, reward_max_idx_aug = reward_batch.max(1)
        reward_aug = reward_max_aug.mean()

        score_no_aug = -reward_no_aug.float()
        score_aug = -reward_aug.float()

        if self.sample_size == 1:
            reward_distances_batch = torch.hstack(torch.split(reward_distances, self.sample_size))
            reward_vehicles_batch = torch.hstack(torch.split(reward_vehicles, self.sample_size))
            reward_depots_batch = torch.hstack(torch.split(reward_depots, self.sample_size))
            nb_depots_batch = torch.hstack(torch.split(nb_depot, self.sample_size))
            nb_vehicles_batch = torch.hstack(torch.split(nb_vehicle, self.sample_size))
            solution_no_aug = solution[0][reward_max_idx][0]
            solution_aug = solution[0][reward_max_idx_aug][0]
            score_dist_aug = -reward_distances_batch[0][reward_max_idx_aug].float()
            score_vehicle_aug = -reward_vehicles_batch[0][reward_max_idx_aug].float()
            score_depot_aug = -reward_depots_batch[0][reward_max_idx_aug].float()
            nb_depots_aug = nb_depots_batch[0][reward_max_idx_aug].float()
            nb_vehicles_aug = nb_vehicles_batch[0][reward_max_idx_aug].float()

            return score_no_aug.item(), score_aug.item(), solution_no_aug, solution_aug, score_dist_aug.item(), score_vehicle_aug.item(), score_depot_aug.item(), nb_depots_aug.item(), nb_vehicles_aug.item()

        
        return score_no_aug.item(), score_aug.item()
    
    def _test_one_batch_bs(self, instance=None):
        aug_type = self.augmentation
        beam_width = self.eval_params['sgbs_beta']     
        expansion_size_minus1 = self.eval_params['sgbs_gamma_minus1']
        rollout_width = beam_width * expansion_size_minus1
        
    
        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_batch_problems(self.sample_size, device=self.device, instance=instance, aug_type=aug_type)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            self.env.depot_mask = self.model.depot_mask


        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, beam_width)
        

        # Beam Search
        ###############################################
        self.env.modify_pomo_size(beam_width)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # the first step
        state, reward, done = self.env.pre_step()
        last_hh = None
        selected, _, last_hh = self.model(state, last_hh)
        # shape: (batch_size, mt_size)
        state, reward, reward_distances, reward_vehicles, reward_depots, done, nb_depot, nb_vehicle = self.env.step(selected)


        # the second step, pomo starting points      
        selected, _, last_hh = self.model(state, last_hh)
        state, reward, reward_distances, reward_vehicles, reward_depots, done, nb_depot, nb_vehicle = self.env.step(starting_points)     


        # BS Step > 1
        ###############################################

        # Prepare Rollout-Env
        rollout_env = copy.deepcopy(self.env)
        rollout_env.modify_pomo_size(rollout_width)
        # self.model.encoded_nodes_mean = self.model.encoded_nodes_mean.repeat(1, expansion_size_minus1, 1)

        # LOOP
        first_rollout_flag = True
        while not done:

            # Next Nodes
            ###############################################
            probs, last_hh = self.model.get_expand_prob(state, last_hh)
            # shape: (aug*batch, beam, problem+1)
            ordered_prob, ordered_i = probs.sort(dim=2, descending=True)

            greedy_next_node = ordered_i[:, :, 0]
            # shape: (aug*batch, beam)

            if first_rollout_flag:
                prob_selected = ordered_prob[:, :, :expansion_size_minus1]
                idx_selected = ordered_i[:, :, :expansion_size_minus1]
                # shape: (aug*batch, beam, rollout_per_node)
            else:
                prob_selected = ordered_prob[:, :, 1:expansion_size_minus1+1]
                idx_selected = ordered_i[:, :, 1:expansion_size_minus1+1]
                # shape: (aug*batch, beam, rollout_per_node)

            # replace invalid index with redundancy
            next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
            is_valid = (prob_selected > 0)
            next_nodes[is_valid] = idx_selected[is_valid]
            # shape: (aug*batch, beam, rollout_per_node)

            # Rollout to get rollout_reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(self.env, repeat=expansion_size_minus1)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later
            last_hh_rollout = last_hh.repeat_interleave(expansion_size_minus1, dim=1)
            next_nodes = next_nodes.reshape(self.env.batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            rollout_state, rollout_reward, reward_distances, reward_vehicles, reward_depots, rollout_done, rollout_nb_depot, roll_out_nb_vehicle = rollout_env.step(next_nodes)
            
            while not rollout_done:
                selected, _, last_hh_rollout = self.model(rollout_state, last_hh_rollout)
                # shape: (batch_size, mt_size)
                rollout_state, rollout_reward, reward_distances, reward_vehicles, reward_depots, rollout_done, rollout_nb_depot, rollout_nb_vehicle = rollout_env.step(selected)

            # rollout_reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(self.env.batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            rollout_reward[is_redundant] = float('-inf')

            # Merge Rollout-Env & BS-Env (Optional, slightly improves performance)
            ##############################################
            # if first_rollout_flag is False:
            #     rollout_env_deepcopy.merge(self.env)
            #     rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
            #     # rollout_reward.shape: (aug*batch, rollout_width + beam_width)
            #     next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
            #     # next_nodes.shape: (aug*batch, rollout_width + beam_width)
            # first_rollout_flag = False

            # BS Step
            ###############################################
            sorted_reward, sorted_index = rollout_reward.sort(dim=1, descending=True)
            beam_reward = sorted_reward[:, :beam_width]
            beam_index = sorted_index[:, :beam_width]
            # shape: (aug*batch, beam_width)

            self.env.reset_by_gathering_rollout_env(rollout_env_deepcopy, gathering_index=beam_index)
            selected = next_nodes.gather(dim=1, index=beam_index)
            # shape: (aug*batch, beam_width)
            # state, reward, done = self.env.step(selected)
            state, reward, reward_distances, reward_vehicles, reward_depots, done, nb_depot, nb_vehicle = self.env.step(selected)

    
        # Return
        ###############################################
        reward_batch = torch.hstack(torch.split(reward, self.sample_size))
        solution = torch.hstack(torch.split(state.selected_node_list, self.sample_size))
        
        
        reward_max, reward_max_idx = reward_batch[:, 0:self.env.depot_size].max(1)
        reward_no_aug = reward_max.mean()
        reward_max_aug, reward_max_idx_aug = reward_batch.max(1)
        reward_aug = reward_max_aug.mean()

        score_no_aug = -reward_no_aug.float()
        score_aug = -reward_aug.float()

        if self.sample_size == 1:
            reward_distances_batch = torch.hstack(torch.split(reward_distances, self.sample_size))
            reward_vehicles_batch = torch.hstack(torch.split(reward_vehicles, self.sample_size))
            reward_depots_batch = torch.hstack(torch.split(reward_depots, self.sample_size))
            nb_depots_batch = torch.hstack(torch.split(nb_depot, self.sample_size))
            nb_vehicles_batch = torch.hstack(torch.split(nb_vehicle, self.sample_size))
            solution_no_aug = solution[0][reward_max_idx][0]
            solution_aug = solution[0][reward_max_idx_aug][0]
            score_dist_aug = -reward_distances_batch[0][reward_max_idx_aug].float()
            score_vehicle_aug = -reward_vehicles_batch[0][reward_max_idx_aug].float()
            score_depot_aug = -reward_depots_batch[0][reward_max_idx_aug].float()
            nb_depots_aug = nb_depots_batch[0][reward_max_idx_aug].float()
            nb_vehicles_aug = nb_vehicles_batch[0][reward_max_idx_aug].float()

            return score_no_aug.item(), score_aug.item(), solution_no_aug, solution_aug, score_dist_aug.item(), score_vehicle_aug.item(), score_depot_aug.item(), nb_depots_aug.item(), nb_vehicles_aug.item()

        
        return score_no_aug.item(), score_aug.item()
    
