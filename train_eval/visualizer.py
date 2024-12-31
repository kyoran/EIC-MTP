import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from train_eval.initialization import initialize_prediction_model, initialize_dataset, get_specific_args
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
import train_eval.utils as u
import imageio
import os
import torch.nn.functional as F

import matplotlib.cm as cm
import matplotlib.patches as mpatches  # 导入用于创建图例的类


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Visualizer:
    """
    Class for visualizing predictions generated by trained model
    """
    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path: str):
        """
        Initialize evaluator object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        # Initialize dataset
        ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
        spec_args = get_specific_args(cfg['dataset'], data_root, cfg['version'] if 'version' in cfg.keys() else None)
        test_set = initialize_dataset(ds_type, ['load_data', data_dir, cfg['test_set_args']] + spec_args)
        self.ds = test_set

        # Initialize model
        self.model = initialize_prediction_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                                 cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(device)
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def visualize(self, output_dir: str, dataset_type: str):
        """
        Generate visualizations for predictions
        :param output_dir: results directory to dump visualizations in
        :param dataset_type: e.g. 'nuScenes'. Visualizations will vary based on dataset.
        :return:
        """
        if dataset_type == 'nuScenes':
            self.visualize_nuscenes(output_dir)

    def visualize_nuscenes(self, output_dir):
        index_list = self.get_vis_idcs_nuscenes()
        if not os.path.isdir(os.path.join(output_dir, 'results', 'gifs')):
            os.mkdir(os.path.join(output_dir, 'results', 'gifs'))
        if not os.path.isdir(os.path.join(output_dir, 'results', 'mp4s')):
            os.mkdir(os.path.join(output_dir, 'results', 'mp4s'))
        for n, indices in enumerate(index_list):
            print("@@@@@@@@@@@@@:", "now at", n)
            imgs = self.generate_nuscenes_gif(indices)
            imgs.append(imgs[-1])
            imgs.append(imgs[-1])

            gif_filename = os.path.join(output_dir, 'results', 'gifs', 'example' + str(n) + '.gif')
            imageio.mimsave(gif_filename, imgs, format='GIF', fps=4)
            mp4_filename = os.path.join(output_dir, 'results', 'mp4s', 'example' + str(n) + '.mp4')
            imageio.mimsave(mp4_filename, imgs, fps=4)


    def get_vis_idcs_nuscenes(self):
        """
        Returns list of list of indices for generating gifs for the nuScenes val set.
        Instance tokens are hardcoded right now. I'll fix this later (TODO)
        """
        token_list = get_prediction_challenge_split('val', dataroot=self.ds.helper.data.dataroot)
        instance_tokens = [token_list[idx].split("_")[0] for idx in range(len(token_list))]
        unique_instance_tokens = []
        for i_t in instance_tokens:
            if i_t not in unique_instance_tokens:
                unique_instance_tokens.append(i_t)  

        # unique_instance_tokens: 789
        # instance_tokens: 9041

        # instance_tokens_to_visualize = [54, 98, 91, 5, 114, 144, 291, 204, 312, 187, 36, 267, 146]
        instance_tokens_to_visualize = list(range(len(unique_instance_tokens))) # 所有都测试

        idcs = []
        for i_t_id in instance_tokens_to_visualize:
            idcs_i_t = [i for i in range(len(instance_tokens)) if instance_tokens[i] == unique_instance_tokens[i_t_id]]

            idcs.append(idcs_i_t)

        return idcs

    def generate_nuscenes_gif(self, idcs: List[int]):
        """
        Generates gif of predictions for the given set of indices.
        :param idcs: val set indices corresponding to a particular instance token.
        """

        # Raster maps for visualization.
        map_extent = self.ds.map_extent
        resolution = 0.1
        layer_names = [
            'drivable_area', 
            # 'ped_crossing', 
            # 'walkway'
        ]
        static_layer_rasterizer = StaticLayerRasterizer(self.ds.helper,
                                                        layer_names=layer_names,
                                                        resolution=resolution,
                                                        meters_ahead=map_extent[3],
                                                        meters_behind=-map_extent[2],
                                                        meters_left=-map_extent[0],
                                                        meters_right=map_extent[1])

        agent_rasterizer = AgentBoxesWithFadedHistory(self.ds.helper, seconds_of_history=1,
                                                      resolution=resolution,
                                                      meters_ahead=map_extent[3],
                                                      meters_behind=-map_extent[2],
                                                      meters_left=-map_extent[0],
                                                      meters_right=map_extent[1])

        raster_maps = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        imgs = []
        for kkkkkkk, idx in enumerate(idcs):

            # Load data
            data = self.ds[idx]
            data = u.send_to_device(u.convert_double_to_float(u.convert2tensors(data)))
            i_t = data['inputs']['instance_token']
            s_t = data['inputs']['sample_token']

            # Get raster map
            hd_map = raster_maps.make_input_representation(i_t, s_t)
            r, g, b = hd_map[:, :, 0] / 255, hd_map[:, :, 1] / 255, hd_map[:, :, 2] / 255
            hd_map_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            # Predict
            inputs = data['inputs']
            predictions, rels = self.model(inputs, is_test=True)

            # Plot
            columns = 7
            fig, ax = plt.subplots(1, columns, figsize=(5*columns, 5))
            ax[0].imshow(hd_map, extent=self.ds.map_extent)
            ax[1].imshow(hd_map_gray, cmap='gist_gray', extent=self.ds.map_extent)
            ax[2].imshow(hd_map_gray, cmap='gist_gray', extent=self.ds.map_extent)

            for n, traj in enumerate(predictions['traj'][0]):
                ax[1].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), lw=4,
                           color='r', alpha=0.8)
                ax[1].scatter(traj[-1, 0].detach().cpu().numpy(), traj[-1, 1].detach().cpu().numpy(), 60,
                              color='r', alpha=0.8)

            traj_gt = data['ground_truth']['traj'][0]
            ax[2].plot(traj_gt[:, 0].detach().cpu().numpy(), traj_gt[:, 1].detach().cpu().numpy(), lw=4, color='g')
            ax[2].scatter(traj_gt[-1, 0].detach().cpu().numpy(), traj_gt[-1, 1].detach().cpu().numpy(), 60, color='g')


            # [ax3] abstract draw
            legend_patchs = []
            alphas = np.linspace(0.3, 1, 5)
            target_agent_feats = inputs['target_agent_representation']  # # [t=5, feat=5]
            nbr_veh_feats = inputs['surrounding_agent_representation']['vehicles']  # # [b=1, max_veh=84, t=5, feat=5]
            nei_veh_masks = inputs['surrounding_agent_representation']['vehicle_masks'] # [b=1, max_veh=84, t=5, 5]    only 0 or 1, 0 is true
            nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
            nei_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks'] # [b=1, max_veh=84, t=5, 5]    only 0 or 1, 0 is true


            # ax[3].imshow(hd_map_gray, cmap='gist_gray', extent=self.ds.map_extent)
            # draw target-agent
            target_pos = target_agent_feats.squeeze().cpu().numpy()
            # ax[3].scatter(target_pos[:, 0], target_pos[:, 1], color='green', alpha=alphas)
            ax[3].scatter(target_pos[-1, 0], target_pos[-1, 1], color='green', alpha=alphas)
            ax[3].annotate('ego', (target_pos[-1, 0] - 0.02, target_pos[-1, 1] - 0.02))

            # ax[3].plot(target_pos[:, 0], target_pos[:, 1], color='green')
            legend_patchs.append(mpatches.Patch(color='green', label='target-agent'))

            # neighbor veh
            nei_veh_masks = ~nei_veh_masks[:, :, :, 0].bool()  # [b=1, max_veh=84, 5]
            nei_veh_masks = nei_veh_masks.any(dim=-1).unsqueeze(2).unsqueeze(3)   # [b=1, max_veh=84, 1, 1]
            nbr_veh_num = nei_veh_masks.sum(axis=1).squeeze().item()
            nbr_veh_feats_batched = torch.masked_select(nbr_veh_feats, nei_veh_masks) # [425]
            nbr_veh_feats_batched = nbr_veh_feats_batched.view(-1, nbr_veh_feats.shape[2], nbr_veh_feats.shape[3])  # (number of all batches, 5, 5)

            # neighbor ped
            nei_ped_masks = ~nei_ped_masks[:, :, :, 0].bool()  # [b=1, max_veh=84, 5]
            nei_ped_masks = nei_ped_masks.any(dim=-1).unsqueeze(2).unsqueeze(3)  # [b=1, max_veh=84, 1, 1]
            nbr_ped_num = nei_ped_masks.sum(axis=1).squeeze().item()
            nbr_ped_feats_batched = torch.masked_select(nbr_ped_feats, nei_ped_masks)  # [425]
            nbr_ped_feats_batched = nbr_ped_feats_batched.view(-1, nbr_ped_feats.shape[2], nbr_ped_feats.shape[3])  # (number of all batches, t=5, 5)

            # draw nei veh+ped
            colors = cm.rainbow(np.linspace(0, 1, nbr_veh_num+nbr_ped_num))
            for one_veh_idx in range(nbr_veh_num):
                one_nbr_veh_feats = nbr_veh_feats_batched[one_veh_idx].cpu().numpy()
                # ax[3].scatter(one_nbr_veh_feats[:, 0], one_nbr_veh_feats[:, 1], color=colors[one_veh_idx], alpha=alphas)
                # ax[3].scatter(one_nbr_veh_feats[:, 0], one_nbr_veh_feats[:, 1], color='blue', alpha=alphas)
                ax[3].scatter(one_nbr_veh_feats[-1, 0], one_nbr_veh_feats[-1, 1], color='blue', alpha=alphas)
                ax[3].annotate('veh'+str(one_veh_idx+1), (one_nbr_veh_feats[-1, 0]+0.02, one_nbr_veh_feats[-1, 1]+0.02))
                # ax[3].plot(one_nbr_veh_feats[:, 0], one_nbr_veh_feats[:, 1], color='blue', alpha=0.2)
                # legend_patchs.append(mpatches.Patch(color=colors[one_veh_idx], label=f'nei-veh-{one_veh_idx}'))

            for one_ped_idx in range(nbr_ped_num):
                one_nbr_ped_feats = nbr_ped_feats_batched[one_ped_idx].cpu().numpy()
                # ax[3].scatter(one_ped_veh_feats[:, 0], one_ped_veh_feats[:, 1], color=colors[one_ped_idx+nbr_veh_num], alpha=alphas)
                # ax[3].scatter(one_ped_veh_feats[:, 0], one_ped_veh_feats[:, 1], color='red', alpha=alphas)
                ax[3].scatter(one_nbr_ped_feats[-1, 0], one_nbr_ped_feats[-1, 1], color='red', alpha=alphas)
                ax[3].annotate('ped' + str(one_ped_idx + 1), (one_nbr_ped_feats[-1, 0] + 0.02, one_nbr_ped_feats[-1, 1] - 0.02))
                # ax[3].plot(one_ped_veh_feats[:, 0], one_ped_veh_feats[:, 1], color='red', alpha=0.2)
                # legend_patchs.append(mpatches.Patch(color=colors[one_ped_idx+nbr_veh_num], label=f'nei-veh-{one_ped_idx}'))
            # import pdb; pdb.set_trace()
            ax[3].set_xlim([self.ds.map_extent[0], self.ds.map_extent[1]])
            ax[3].set_ylim([self.ds.map_extent[2], self.ds.map_extent[3]])
            # ax[3].legend(handles=legend_patchs, loc='upper right')
            ax[3].legend(loc='upper right')



            # mimg = ax[4].imshow(rels[0][0][0, 0].squeeze().cpu().detach().numpy(), cmap='YlGnBu')
            # fig.colorbar(mimg, ax=ax[4])
            # ax[4].set_title("explicit_t_0")

            # mimg = ax[5].imshow(rels[0][0][0, 4].squeeze().cpu().detach().numpy(), cmap='YlGnBu')
            # fig.colorbar(mimg, ax=ax[5])
            # ax[5].set_title("explicit_t_4")

            # [Explicit]
            # 计算每列的和
            exp_matrix = rels[1][0]
            # col_sums = torch.sum(exp_matrix, dim=0, keepdim=True)
            # exp_matrix = exp_matrix / col_sums  # 对每列进行softmax
            exp_matrix = exp_matrix.squeeze().cpu().detach().numpy()
            mimg = ax[4].imshow(exp_matrix, cmap='YlGnBu')
            fig.colorbar(mimg, ax=ax[4])
            ax[4].text(0.5, -0.1,  # 0.5表示水平居中，0表示最下方
                       f"nei_veh={nbr_veh_num}, nei_ped={nbr_ped_num}",
                       verticalalignment='bottom', horizontalalignment='center', transform=ax[4].transAxes)
            ax[4].set_title("Explicit collaboration")

            # [Implicit]
            imp_matrix = rels[2][0]
            # imp_matrix = F.softmax(imp_matrix, dim=-2)
            imp_matrix = imp_matrix.squeeze().cpu().detach().numpy()
            mimg = ax[5].imshow(imp_matrix, cmap='YlGnBu')
            fig.colorbar(mimg, ax=ax[5])
            ax[5].set_title("Implicit collaboration")
            
            # [Semantic]
            mimg = ax[6].imshow(rels[3][0].squeeze().cpu().detach().numpy(), cmap='YlGnBu')
            fig.colorbar(mimg, ax=ax[6])
            ax[6].set_title("Semantic collaboration")

            # style
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[3].axis('off')
            ax[4].axis('off')
            ax[5].axis('off')
            ax[6].axis('off')

            fig.tight_layout(pad=0)
            ax[0].margins(0)
            ax[1].margins(0)
            ax[2].margins(0)
            ax[3].margins(0)
            ax[4].margins(0)
            ax[5].margins(0)
            ax[6].margins(0)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            imgs.append(image_from_plot)
            plt.close(fig)
            plt.cla()
            plt.clf()


        return imgs
