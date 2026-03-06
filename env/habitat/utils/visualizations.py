import sys

import matplotlib
import numpy as np

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
import skimage
from einops import rearrange

# define colors in hex
eva_purple1 = "#BB84EBF2"
eva_purple2 = "#5549B7" 
eva_darkpurple = "#201D30" 
eva_green = "#8EDF5F" 
eva_orange = "#EC7744" 
rei_white = "#E1F6F8" 
rei_white2 = "#CDD3F4" 
rei_blue = "#25629B" 
elster_red = "#93092b" 
eva02_red = '#ed2323'

desc = ['path', 'gt_path', 'collision', 'pred obstacle', 'explorable', 'explored', 'global', 'local' ]
color_palette = sns.color_palette([rei_white, rei_white2, eva_darkpurple, eva_green, eva_purple1, eva_orange, eva02_red, rei_blue])


def visualize(fig, ax, 
              img, grid, uncert_map,
              pos, gt_pos, 
              dump_dir, rank, ep_no, t,
              visualize, print_images, previous_action, accumulated_ratio,
              heuristic_active=None,
              uncert_sum_history=None,
              reward_history=None,
              value_history=None,
              uncert_init=1e-6,
              gt_map=None,
              full_map=None,
              local_map=None):
    """
    Args:
        rank: Thread No.
        ep_no: current episode
        t: time step
        accumulated_ratio: percentage of map exlored
    """
    if uncert_map is None:
        for i in range(2):
            ax[i].clear()
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])

        ax[0].imshow(img)
        ax[0].set_title(f"Pre_Act={previous_action}", fontsize=15, color='red' if heuristic_active else 'black')

        title = f"Step={t}, Exp_ratio={accumulated_ratio:.2f}"
        if heuristic_active is not None:
            title += f", Heuristic={'On' if heuristic_active else 'Off'}"

        ax[1].imshow(grid, origin='lower') # to be right hand coordinate
        ax[1].set_title(title, fontsize=15, color='red' if heuristic_active else 'black')
    else:
        for i in range(len(ax)):
            ax[i].clear()

        ax[0].imshow(img)
        ax[0].set_title(f"Pre_Act={previous_action}", fontsize=15)

        title = f"Step={t}, Exp_ratio={accumulated_ratio:.2f}"
 
        ax[1].imshow(grid, origin='lower') # to be right hand coordinate
        ax[1].set_title(title, fontsize=15)

        ax[2].imshow(uncert_map, origin='lower', cmap='plasma', vmin=0, vmax=uncert_init) # to be right hand coordinate
        if gt_map is not None:
            ax[2].contour(gt_map, levels=[0.5], colors='grey', linewidths=1, origin='lower')
        
        ax[2].set_title('Uncertainty_map', fontsize=15)

        # Plot Uncertainty History
        if len(uncert_sum_history)>0:
            steps, sums = zip(*uncert_sum_history)
            ax[3].plot(steps, sums, marker='o')
        ax[3].set_title("Uncertainty Mean vs Step")
        ax[3].set_xlabel("Step")
        ax[3].set_ylabel("Uncertainty Mean")
        ax[3].grid(True)

        # Plot Reward History
        if len(reward_history)>0:
            steps, rewards = zip(*reward_history)
            ax[4].plot(steps, rewards, marker='o', color='green')
        ax[4].set_title("Reward vs Step")
        ax[4].set_xlabel("Step")
        ax[4].set_ylabel("Reward")
        ax[4].grid(True)

        # Plot Value History
        if value_history is not None and len(value_history)>0:
            steps, values = zip(*value_history)
            ax[5].plot(steps, values, marker='o', color='orange')
        ax[5].set_title("Value vs Step")
        ax[5].set_xlabel("Step")
        ax[5].set_ylabel("Value")
        ax[5].grid(True)

        if len(ax) > 6 and full_map is not None:
            # Normalize the second channel (index 1)
            full_map_to_draw = full_map.copy()
            f_max = full_map_to_draw[1].max()
            if f_max > 0:
                full_map_to_draw[1] /= f_max
            # Concatenate the four channels along width dimension
            full_map_cat = rearrange(full_map_to_draw, 'c h w -> h (c w)')
            ax[6].imshow(full_map_cat, origin='lower')
            ax[6].set_title('Full Map Channels (Obs, Exp/Unc, Agent, Visited)')

        if len(ax) > 7 and local_map is not None:
            # Normalize the second channel (index 1)
            local_map_to_draw = local_map.copy()
            l_max = local_map_to_draw[1].max()
            if l_max > 0:
                local_map_to_draw[1] /= l_max
            # Concatenate the four channels along width dimension
            local_map_cat = rearrange(local_map_to_draw, 'c h w -> h (c w)')
            ax[7].imshow(local_map_cat, origin='lower')
            ax[7].set_title('Local Map Channels (Obs, Exp/Unc, Agent, Visited)')

    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    #x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0
    x, y = x * 100.0 / 5.0, y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    # Draw predicted agent pose
    x, y, o = pos
    #x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0
    x, y = x * 100.0 / 5.0, y * 100.0 / 5.0
    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)

    legend_elements = [
        mpatches.Patch(color=color_palette[i], label=desc[i]) \
        for i in range(len(color_palette))
    ]
    ax[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                 loc='upper left', borderaxespad=0., fontsize=8)

    for _ in range(5):
        plt.tight_layout()

    if visualize:
        print("vis!")
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/thread_{}/ep_{}/{:04d}.png'.format(
            dump_dir, rank+1, ep_no, t)
        plt.savefig(fn)


def fill_color(colored, map, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - map)
        colored[:, :, 2 - i] += (1 - color[i]) * map
    return colored


def get_colored_map(map, collision_map, visited, visited_gt, goal, local_goal,
                    explored, gt_map, gt_map_explored):
    """
        @param map: predicted map
        @param collision_map: collision points along the map
        @param visited: predicted visited path
        @param visited_gt: gt visited path
        @param goal: long term goal from global policy 
        @param local_goal: local goal from planner 
        @param explored: gt explored map 
        @param gt_map: total explorable map 
        @param gt_map_explored: redundant? 
    """
    m, n = map.shape
    colored = np.zeros((m, n, 3))

    colored = fill_color(colored, gt_map, color_palette[4])
    colored = fill_color(colored, explored, color_palette[5])
    colored = fill_color(colored, map, color_palette[3])
    colored = fill_color(colored, visited_gt, color_palette[1])
    colored = fill_color(colored, visited, color_palette[0])
    colored = fill_color(colored, collision_map, color_palette[2])
    
    
    #colored = fill_color(colored, gt_map_explored, color_palette[3])
    
    

    # plot global goal 
    selem = skimage.morphology.disk(4)
    goal_map = np.zeros((m, n))
    goal_map[goal[1], goal[0]] = 1
    goal_map = 1 - skimage.morphology.binary_dilation(
        goal_map, selem) != True

    colored = fill_color(colored, goal_map, color_palette[6])

    # plot local goal
    selem = skimage.morphology.disk(4)
    local_goal_map = np.zeros((m, n))
    local_goal_map[int(local_goal[1]), int(local_goal[0])] = 1
    local_goal_map = 1 - skimage.morphology.binary_dilation(
        local_goal_map, selem) != True
    colored = fill_color(colored, local_goal_map, color_palette[7])


    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored