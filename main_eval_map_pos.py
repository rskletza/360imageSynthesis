import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon

import evaluation, evaluation_sets
import utils

res_sets = evaluation_sets.pos_res_sets
plot_types = evaluation.METRICTYPES


min_vals = np.full(len(plot_types), np.inf)
max_vals = np.full(len(plot_types), 0.0)

#set the colors
for res_set in res_sets:
    for p_type in plot_types:
        i = plot_types.index(p_type)
        vals = np.array(res_set.get_metrics_by_type(p_type))
        min_val = np.amin(vals)
        max_val = np.amax(vals)
        if min_val < min_vals[i]:
            min_vals[i] = min_val
        if max_val > max_vals[i]:
            max_vals[i] = max_val

print(min_vals)
print(max_vals)

for res_set in res_sets:
    fig, ax = plt.subplots(1,len(plot_types), figsize=(30,10))
    fig.suptitle(res_set.get_name(), fontsize=28, weight='bold')
    image = utils.load_img(res_set.tlpath + "../" + "top_gray.jpg")

    gt_pos = res_set.get_gt_positions()
    vps = res_set.get_vps()

    for i, p_type in enumerate(plot_types):
        if image is not None:
            dims = res_set.dims
            ax[i].imshow(image, extent=(-dims[0]/2, dims[0]/2, -dims[1]/2, dims[1]/2))
        vals = np.array(res_set.get_metrics_by_type(p_type))
        norm_vals = (vals - min_vals[i]) / (max_vals[i]-min_vals[i])
        cmap = matplotlib.cm.get_cmap("coolwarm")
        colors = cmap(norm_vals)
        ax[i].scatter(gt_pos[:,0], gt_pos[:,1], color=colors, s=100)
        ax[i].set_title(p_type)
#    ax[i].scatter(vps[:,0], vps[:,1], color='black', marker = 'x')
        for j, p in enumerate(gt_pos):
            ax[i].annotate(res_set.ids[j], xy=(p[:2]), xytext=(2, 2), textcoords='offset points')

    plt.savefig(utils.OUT + "eval_posmap_normalized" + res_set.get_name() + '.png', bbox_inches='tight', dpi=100)#utils.DPI)
    plt.show()
