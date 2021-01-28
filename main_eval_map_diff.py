import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

import evaluation, evaluation_sets
import utils

res_sets = evaluation_sets.pos_res_sets
plot_types = evaluation.USED_METRICTYPES

plt.rcParams.update({'font.size': 24})

if len(res_sets) != 2:
    raise Exception("Only two result sets can be compared. " + len(res_sets) + " were passed.")

first = res_sets[0]
second = res_sets[1]

fig, ax = plt.subplots(len(plot_types),1, figsize=(10,20))
#fig.suptitle("Improvement of\n"+second.get_blendtype() +" over " + first.get_blendtype() + " blending", fontsize=28, weight='bold', x=0.57)
#fig.suptitle("Improvement of a density of\n 6x6 over 2x2 using " + first.get_blendtype() + " blending", fontsize=28, weight='bold', x=0.57)
image = utils.load_img(first.tlpath + "../" + "top_gray.jpg")

#other
gt_pos = first.get_gt_positions()
vps = first.get_vps()

#sw
#gt_pos_switch = first.get_gt_positions()
#gt_pos = np.dstack((gt_pos_switch[:,1], gt_pos_switch[:,0], gt_pos_switch[:,2]))[0] * np.array([1,-1,1])
#vps_switch = first.get_vps()
#vps = np.dstack((vps_switch[:,1], vps_switch[:,0], vps_switch[:,2]))[0] * np.array([1,-1,1])


cmap = matplotlib.cm.get_cmap("coolwarm")
blue = cmap(0)
red = cmap(-1)
for i, p_type in enumerate(plot_types):
    if image is not None:
        dims = first.dims
        ax[i].imshow(image, extent=(-dims[0]/2, dims[0]/2, -dims[1]/2, dims[1]/2))
        ax[i].axis('off')
    first_vals = np.array(first.get_metrics_by_type(p_type))
    second_vals = np.array(second.get_metrics_by_type(p_type))
    diff_vals = second_vals - first_vals

    min_val = np.argmin(diff_vals)
    max_val = np.argmax(diff_vals)
    print(p_type, ": best: ", min_val, ", worst: ", max_val)
    print("best improvement: ", np.amin(diff_vals))
    print("worst 'improvement': ", np.amax(diff_vals))
    print(np.argsort(diff_vals))
    better = 0
    worse = 0
    same = 0
    for val in diff_vals:
        if val < 0:
            better += 1
        elif val > 0:
            worse += 1
        else:
            same += 1
    print("better: ", better)
    print("worse: ", worse)
    print("same: ", same)
    print("sum: ", better + worse + same)

    abs_max_val = np.amax(np.abs(diff_vals))
    norm_vals = (diff_vals + abs_max_val) / (2*abs_max_val)

    colors = cmap(norm_vals)
    norm = matplotlib.colors.Normalize(vmin=-abs_max_val, vmax=abs_max_val)

    #regular scenes
#    sc = ax[i].scatter(gt_pos[:,0], gt_pos[:,1], color=colors, s=400, cmap=cmap, edgecolors="black")
#   #annotate s_points
#    for j, p in enumerate(gt_pos):
#        ax[i].annotate(first.ids[j], xy=(p[:2]), xytext=(2, 2), textcoords='offset points')
#    ax[i].annotate("\u2605", xy=(gt_pos[max_val][:2]), xytext=(15,2), textcoords='offset pixels', color="black")
#    ax[i].annotate("\u2605", xy=(gt_pos[min_val][:2]), xytext=(15,2), textcoords='offset pixels', color="black")

    divider = make_axes_locatable(ax[i])
    caxes = divider.append_axes("right", size='5%', pad=0.2)
    round_abs_max = np.round(abs_max_val, 3)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),orientation='vertical',ax=ax[i], cax=caxes, ticks=[-round_abs_max+0.001, 0, round_abs_max-0.001])

    #dense scenes
    sc = ax[i].scatter(gt_pos[:,0], gt_pos[:,1], color=colors, s=300, cmap=cmap, edgecolors="black")
   #mark best and worst net values
#    ax[i].text(gt_pos[max_val][0], gt_pos[max_val][1], s="\u2605", color="black", verticalalignment="center", horizontalalignment="center", fontsize="xx-large", weight="bold")
#    ax[i].text(gt_pos[max_val][0], gt_pos[max_val][1], s="\u2605", color=cmap(max_val), verticalalignment="center", horizontalalignment="center", fontsize="large", weight="bold")
#    ax[i].text(gt_pos[min_val][0], gt_pos[min_val][1], s="\u2605", color="black", verticalalignment="center", horizontalalignment="center", fontsize="xx-large", weight="bold")
#    ax[i].text(gt_pos[min_val][0], gt_pos[min_val][1], s="\u2605", color=cmap(-1), verticalalignment="center", horizontalalignment="center", fontsize="large", weight="bold")



    ax[i].set_title("\N{Greek Capital Letter Delta}" + evaluation.get_metric_name(p_type))
    #draw captured points
    ax[i].scatter(vps[:,0], vps[:,1], color='black', marker = 'X', s=100)

    #square room
    ax[i].text(2.07,-1.0,"better")
    ax[i].text(2.07,1.0,"worse")
    #oblong room
#    ax[i].text(3.3,-1.0,"better")
#    ax[i].text(3.3,1.0,"worse")
    #sw
#    ax[i].text(164,-110.0,"better")
#    ax[i].text(164,110.0,"worse")


        #annotate values of s_points
#            ax[i].annotate(np.round(vals[j],3), xy=(p[:2]), xytext=(2, -6), textcoords='offset points')

fig.subplots_adjust(wspace=0.0, hspace=0.2)
#border = plt.Rectangle(
#        (0.17,0.1), 0.83, 0.81, fill=False, color="k", lw=4,
#        zorder=1000, transform=fig.transFigure, figure=fig
#)
border = plt.Rectangle(
        (0.1,0.1), 0.94, 0.81, fill=False, color="k", lw=4,
        zorder=1000, transform=fig.transFigure, figure=fig
)
fig.patches.extend([border])

plt.savefig(utils.OUT + "diff_" + first.scene + "_" + first.get_blendtype() + '-' + second.get_blendtype() + '.png', bbox_inches='tight', dpi=75)#utils.DPI)
#plt.show()
