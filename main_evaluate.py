import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import string

import evaluation
import evaluation_sets
import utils
'''
each result has its own folder (to store evaluation in as well)
id_out_flow_params.jpg: the output image, params like density, viewpoints etc
id_out_reg_params
id_out_baseline_params.jpg: the baseline image

will create:
id_eval_params.jpg: evaluation output images, params for which evaluation it is
id_eval.txt: for the error values
'''
'''
paramstring variations
dens4, dens8, dens16
dist1, dist2, dist3, dist4
vp2, vpr
'''
plt.rcParams.update({'font.size': 18})

res_sets = evaluation_sets.res_sets

plot_types = evaluation.USED_METRICTYPES

fig, ax = plt.subplots(1,len(plot_types), figsize=(18,15))
#fig, ax = plt.subplots(1,len(plot_types), figsize=(10,5))
#fig.suptitle(evaluation_sets.name, fontsize=28, weight='bold')

#where will the points be placed on the x axis
#leave a space between each res_set
y = []
n = 0
for i in range(len(res_sets)):
    for j in range(len(res_sets[i])):
        n += 1
        y.append(n)
    n += 1

y = -1 * np.array(y)

boxplot_data = [[],[],[]]
boxplot_color = []
boxplot_labels = []

#retrieve the data and parameters
for i, x_comp in enumerate(res_sets): #compare by placing close on x axis
    for res_set in x_comp:
        boxplot_color.append(res_set.get_color())
        boxplot_labels.append(res_set.get_name())
        for j in range(len(plot_types)):
            boxplot_data[j].append(res_set.get_metrics_by_type(plot_types[j]))

#draw the boxplots
for i, plot_type in enumerate(plot_types):
    bp = ax[i].boxplot(boxplot_data[i], sym='+', positions=y, notch=0, medianprops = dict(linestyle='-', linewidth=4, color='black'), showmeans=0, meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='none'), vert=False)

    #handle each plot
    for plotnum, plot in enumerate(boxplot_data[i]):
        #label the best and worst values
#        ax[i].annotate(string.ascii_uppercase[np.argmin(plot)], xy=(x[plotnum], np.amin(plot)), xytext=(2, 2), textcoords='offset points', weight='bold')
#        ax[i].annotate(string.ascii_uppercase[np.argmax(plot)], xy=(x[plotnum], np.amax(plot)), xytext=(2, 2), textcoords='offset points', weight='bold')

        #fill the boxes with the corresponding colors
        #from (https://matplotlib.org/3.3.3/gallery/statistics/boxplot_demo.html)
        box = bp['boxes'][plotnum]
        box_x = []
        box_y = []
        for j in range(5): #5 for boxes, 11 for notched
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax[i].add_patch(Polygon(box_coords, facecolor=boxplot_color[plotnum], linewidth=5, alpha=0.6))

    ax[i].xaxis.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.5)
    if i % 2 != 0:
        ax[i].yaxis.set_ticks_position("right")
    ax[i].set(
        axisbelow = True, #hide grid behind plot items
        title = evaluation.get_metric_name(plot_type),
        yticks = y,
            )
    if plot_type == "rgb_l1":
        ax[i].set_title("L1 error")
    else:
        ax[i].set_title("SSIM error")
    ax[i].set_yticklabels(boxplot_labels)

custom_lines = []
custom_labels = []
for color in set(boxplot_color):
    custom_lines.append(Line2D([0], [0], color=color, lw=4))
    if evaluation.get_blendtype_by_color(color) == evaluation.BLENDTYPES[0]:
        custom_labels.append("naive algorithm")
    else:
        custom_labels.append(evaluation.get_blendtype_by_color(color) + " blending")
fig.legend(custom_lines, custom_labels, loc="lower center", borderaxespad=-0.2)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig(utils.OUT + evaluation_sets.name + '.png', bbox_inches='tight', dpi=100)#utils.DPI)
plt.show()
