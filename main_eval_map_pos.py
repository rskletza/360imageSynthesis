import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from palettable.colorbrewer.sequential import GnBu_8_r as colmap
#from palettable.cmocean.sequential import Thermal_5 as colmap

import evaluation, evaluation_sets
import utils

res_sets = evaluation_sets.pos_res_sets
plot_types = evaluation.USED_METRICTYPES

plt.rcParams.update({'font.size': 24})

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

cmap = matplotlib.cm.get_cmap(colmap.mpl_colormap)
for res_set in res_sets:
    fig, ax = plt.subplots(len(plot_types),1, figsize=(10,20))
#    fig.suptitle(res_set.get_name(), fontsize=28, weight='bold')
    image = utils.load_img(res_set.tlpath + "../" + "top_gray.jpg")

    #sw
#    gt_pos_switch = res_set.get_gt_positions()
#    gt_pos = np.dstack((gt_pos_switch[:,1], gt_pos_switch[:,0], gt_pos_switch[:,2]))[0] * np.array([1,-1,1])
#    vps_switch = res_set.get_vps()
#    vps = np.dstack((vps_switch[:,1], vps_switch[:,0], vps_switch[:,2]))[0] * np.array([1,-1,1])

    #other
    gt_pos = res_set.get_gt_positions()
    vps = res_set.get_vps()


    for i, p_type in enumerate(plot_types):
        vals = np.array(res_set.get_metrics_by_type(p_type))
        norm_vals = (vals - min_vals[i]) / (max_vals[i]-min_vals[i])
        if image is not None:
            dims = res_set.dims
            ax[i].imshow(image, extent=(-dims[0]/2, dims[0]/2, -dims[1]/2, dims[1]/2))
            ax[i].axis('off')
#            colorbar = matplotlib.colorbar.ColorbarBase(ax[1,i], norm=norm, cmap=cmap, orientation='horizontal')
#            fig.colorbar(vals.reshape, ax=ax[i], cmap=cmap, norm=norm, orientation='horizontal')

        colors = cmap(norm_vals)
        norm = matplotlib.colors.Normalize(vmin=min_vals[i], vmax=max_vals[i])
#        sc = ax[i].scatter(gt_pos[:,0], gt_pos[:,1], color=colors, s=200, cmap=cmap, edgecolors="black")
        sc = ax[i].scatter(gt_pos[:,0], gt_pos[:,1], color=colors, s=600, cmap=cmap, edgecolors="black")
        divider = make_axes_locatable(ax[i])
        caxes = divider.append_axes("right", size='5%', pad=0.2)
        round_min = np.round(min_vals[i], 3)+0.001
        round_max = np.round(max_vals[i], 3)-0.001
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),orientation='vertical',ax=ax[i], cax=caxes, ticks=[round_min, round_max])
        #square room
        ax[i].text(2.05,-1.0,"better")
        ax[i].text(2.05,1.0,"worse")
        #oblong room
#        ax[i].text(3.3,-1.0,"better")
#        ax[i].text(3.3,1.0,"worse")
        #sw
#        ax[i].text(155,-100.0,"better")
#        ax[i].text(155,100.0,"worse")


        ax[i].set_title(evaluation.get_metric_name(p_type))
        #draw captured points
#        ax[i].scatter(vps[:,0], vps[:,1], color='black', marker = 'X', s=80)
#        ax[i].scatter(vps[:,0], vps[:,1], color='black', marker = 'x', s=200)

        #annotate s_points
        for j, p in enumerate(gt_pos):
            ax[i].annotate(res_set.ids[j], xy=(p[:2]), xytext=(2, 2), textcoords='offset points')
            #annotate values of s_points
#            ax[i].annotate(np.round(vals[j],3), xy=(p[:2]), xytext=(2, -6), textcoords='offset points')

#    fig.subplots_adjust(wspace=0.0, hspace=0.2)
#    border = plt.Rectangle(
#            (0.16,0.1), 0.80, 0.81, fill=False, color="k", lw=4,
#            zorder=1000, transform=fig.transFigure, figure=fig
#    )
    border = plt.Rectangle(
            (0.1,0.1), 0.92, 0.81, fill=False, color="k", lw=4,
            zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([border])

    plt.savefig(utils.OUT + res_set.scene + "_" + res_set.pos + "_" + evaluation.BLENDTYPES[res_set.blendtype] + '.png', bbox_inches='tight', dpi=50)#utils.DPI)
#    plt.show()
