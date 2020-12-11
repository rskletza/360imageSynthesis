import json
import numpy as np
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
res_sets = evaluation_sets.res_sets

plot_types = evaluation.METRICTYPES

fig, ax = plt.subplots(1,len(plot_types), figsize=(25,12))
fig.suptitle(evaluation_sets.name, fontsize=28, weight='bold')

#where will the points be placed on the x axis
#leave a space between each res_set
x = []
n = 0
for i in range(len(res_sets)):
    for j in range(len(res_sets[i])):
        n += 1
        x.append(n)
    n += 1

boxplot_data = [[],[],[]]
boxplot_color = []
boxplot_labels = []

#retrieve the data and parameters
for i, x_comp in enumerate(res_sets): #compare by placing close on x axis
    for res_set in x_comp:
        boxplot_color.append(res_set.get_color())
        boxplot_labels.append(res_set.get_name())
        boxplot_data[0].append(res_set.get_metrics_by_type(evaluation.METRICTYPES[0]))
        boxplot_data[1].append(res_set.get_metrics_by_type(evaluation.METRICTYPES[1]))
        boxplot_data[2].append(res_set.get_metrics_by_type(evaluation.METRICTYPES[2]))


#draw the boxplots
for i, plot_type in enumerate(plot_types):
    bp = ax[i].boxplot(boxplot_data[i], sym='+', positions=x, notch=0, medianprops = dict(linestyle='-', linewidth=2.5, color='black'), showmeans=0, meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='none'))

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

    ax[i].yaxis.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.5)
    ax[i].set(
        axisbelow = True, #hide grid behind plot items
        title = plot_type,
        ylabel = 'error',
        xticks = x,
            )
    ax[i].set_xticklabels(boxplot_labels, rotation=45)


plt.savefig(utils.OUT + "eval_" + evaluation_sets.name + '.png', bbox_inches='tight', dpi=utils.DPI)
plt.show()


