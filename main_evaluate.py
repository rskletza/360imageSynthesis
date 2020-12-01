import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import evaluation
import utils
'''
run one evaluation pass
1.
current density, current virtual scene
all viewpoints within 2x min density distance
only deviation angle based

2.
also flow-based

need functions to calculate distance:
check for cubemap representation
L2 & L1 for RGB and grayscale
L1 for edges --> which edge filter
SSIM

need system to keep track of results
x points on trajectory
track parameters in filename
store graphs of
- best 2 deviation angles
- indices used
- interpolation distances
- index distances

functions to calculate baseline interpolation --> in interpolator class

'''

#capture set to evaluate
#cap_set = ""
#ground truth folder (unique id for each image)
#gt_dir = cap_set.location + "ground_truth/"
#'gt_id.jpg'

#calculate results in folder eval_params (same ids as in ground truth folder)
'''
each result has its own folder (to store evaluation in as well)
id_out_flow_params.jpg: the output image, params like density, viewpoints etc
id_out_reg_params
id_out_baseline_params.jpg: the baseline image

will create:
id_eval_params.jpg: evaluation output images, params for which evaluation it is
id_eval.txt: for the error values
'''
eval_dir = "../../data/eval/test/"
out_dir = "../../data/captures/synthesized_room/square_room_textured_brick_small/evaluation/"
gt_dir = "../../data/captures/synthesized_room/square_room_textured_brick_small/ground_truth/"
ids = list(map(str, [8, 10, 12, 16, 18, 22, 24, 26, 30, 32, 36, 38, 40]))
imtypes = ["rgb", "gray", "edges"]
restypes = ["baseline", "reg", "flow"]
paramstring = "dens7_dist2_vp2"
'''
paramstring variations
dens4, dens8, dens16
dist1, dist2, dist3, dist4
vp2, vpr
'''

try:
    with open(eval_dir + 'full_eval.npy', 'rb') as f:
        eval_vals_np = np.load(f)

except(FileNotFoundError):
    full_eval = {}
    eval_vals_np = np.full((len(ids), 9), np.Inf)

    for i, id in enumerate(ids):
        #create a folder with this id name in the eval_dir
        os.mkdir(eval_dir + id)

        #for each output image ..
        img_reg = evaluation.prep_image(cv2.cvtColor(cv2.imread(out_dir + id + "_out.jpg", 1), cv2.COLOR_BGR2RGB)) #the 'regular' output image with deviation-angle-based blending

        img_flow = evaluation.prep_image(cv2.cvtColor(cv2.imread(out_dir + id + "_out_flow.jpg", 1), cv2.COLOR_BGR2RGB)) #the output image with flow-based blending

        img_baseline = evaluation.prep_image(cv2.cvtColor(cv2.imread(out_dir + id + "_baseline.jpg", 1), cv2.COLOR_BGR2RGB))  #the baseline image

        gt = evaluation.prep_image(cv2.cvtColor(cv2.imread(gt_dir + id + ".jpg"), cv2.COLOR_BGR2RGB))

        imgs = {
                    "rgb":
                    {
                        "baseline": img_baseline,
                        "reg": img_reg,
                        "flow": img_flow,
                        "gt": gt
                    },
                    "gray": 
                    {
                        "baseline": cv2.cvtColor(img_baseline, cv2.COLOR_RGB2GRAY),
                        "reg": cv2.cvtColor(img_reg, cv2.COLOR_RGB2GRAY),
                        "flow": cv2.cvtColor(img_flow, cv2.COLOR_RGB2GRAY),
                        "gt": cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
                    }
                }
        imgs["edges"] = {
                        "baseline": cv2.Laplacian(cv2.GaussianBlur(imgs["gray"]["baseline"], (3,3),0), cv2.CV_32F),
                        "reg": cv2.Laplacian(cv2.GaussianBlur(imgs["gray"]["reg"], (3,3),0), cv2.CV_32F),
                        "flow": cv2.Laplacian(cv2.GaussianBlur(imgs["gray"]["flow"], (3,3),0), cv2.CV_32F),
                        "gt": cv2.Laplacian(cv2.GaussianBlur(imgs["gray"]["gt"], (3,3),0), cv2.CV_32F)
                    }
        eval_vals = {
                        "baseline":
                        {
                            "rgb_l1": np.Inf, "gray_ssim": np.Inf, "edge_l2": np.Inf
                        },
                        "reg":
                        {
                            "rgb_l1": np.Inf, "gray_ssim": np.Inf, "edge_l2": np.Inf
                        },
                        "flow":
                        {
                            "rgb_l1": np.Inf, "gray_ssim": np.Inf, "edge_l2": np.Inf
                        }
        }

        #TODO is the faulty conversion from cube to latlong actually an issue in error eval because evaluation is done in cubemap anyway?

        for pos, (restype, eval_types) in enumerate(eval_vals.items()):
            s = imgs["rgb"][restype]
            gt = imgs["rgb"]["gt"]
            error, vis = evaluation.l1_error(gt, s)
            eval_vals[restype]["rgb_l1"] = error
            eval_vals_np[i, pos*3 + 0] = error
            utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_l1.jpg', eval_dir + id + "/")

            s = imgs["gray"][restype]
            gt = imgs["gray"]["gt"]
            error, vis = evaluation.ssim(gt, s)
            eval_vals[restype]["gray_ssim"] = error
            eval_vals_np[i, pos*3 + 1] = error
#        utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_ssim.jpg', eval_dir + id + "/")

            s = imgs["edges"][restype]
            gt = imgs["edges"]["gt"]
            error, vis = evaluation.l2_error(gt, s)
            eval_vals[restype]["edge_l2"] = error
            eval_vals_np[i, pos*3 + 2] = error
            utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_l2.jpg', eval_dir + id + "/")
        
    with open(eval_dir + '/full_eval.npy', 'wb') as f:
        np.save(f, eval_vals_np)

#plot graphs (colors from https://venngage.com/blog/color-blind-friendly-palette/#4)
color1 = '#f5793a'
color2 = '#a95aa1'
color3 = '#85c0f9' #'#0f2080'

fig, ax = plt.subplots(1,3, figsize=(25,12))
x = np.array(list(range(len(ids))))

ax[0].plot(x, eval_vals_np[:,0], color=color1, label='baseline', linestyle='--') #baseline rgb
ax[0].plot(x, eval_vals_np[:,3], color=color2, label='regular blending', linestyle='-') #reg rgb
ax[0].plot(x, eval_vals_np[:,6], color=color3, label='flow-based blending', linestyle='-.') #flow rgb
ax[0].set_title('L1 error on RGB images')
ax[0].set_xlabel('viewpoint index')
ax[0].set_ylabel('error')
ax[0].set_xticks(x)
ax[0].set_xticklabels(ids)

ax[1].plot(x, 1 - eval_vals_np[:,1], color=color1, linestyle='--') #baseline edge
ax[1].plot(x, 1 - eval_vals_np[:,4], color=color2, linestyle='-') #reg edge
ax[1].plot(x, 1 - eval_vals_np[:,7], color=color3, linestyle='-.') #flow edge
ax[1].set_title('Inverted SSIM measure on grayscale images (1 - ssim)')
ax[1].set_xlabel('viewpoint index')
ax[1].set_ylabel('error')
ax[1].set_xticks(x)
ax[1].set_xticklabels(ids)

ax[2].plot(x, eval_vals_np[:,2], color=color1, linestyle='--') #baseline gray
ax[2].plot(x, eval_vals_np[:,5], color=color2, linestyle='-') #reg gray
ax[2].plot(x, eval_vals_np[:,8], color=color3, linestyle='-.') #flow gray
ax[2].set_title('L2 error on edges (Laplace filter on smoothed image)')
ax[2].set_xlabel('viewpoint index')
ax[2].set_ylabel('error')
ax[2].set_xticks(x)
ax[2].set_xticklabels(ids)

fig.legend()

plt.savefig(eval_dir + 'eval_graph.png', bbox_inches='tight')
plt.show()
plt.clf()
'''
    json file
    eval_vals = {
                    "rgb":
                    {   
                        "baseline": {}, "reg": {}, "flow": {}
                    },
                    "gray":
                    {   
                        "baseline": {}, "reg": {}, "flow": {}
                    },
                    "edges":
                    {
                        "baseline": {}, "reg": {}, "flow": {}
                    }
                }
#for rgb, grayscale, edges
    for imtype, restypes in eval_vals.items():
        print(imtype, restypes)
        #for baseline, regular, and flow (result type)
        for restype in restypes:
            s = imgs[imtype][restype]
            gt = imgs[imtype]["gt"]
            error, vis = evaluation.l1_error(gt, s)
            eval_vals[imtype][restype]["l1"] = error
            utils.cvwrite(vis, 'l1_er_' + restype + '_' + imtype + '.jpg', eval_dir + id + "/")

            error, vis = evaluation.l2_error(gt, s)
            eval_vals[imtype][restype]["l2"] = error
            utils.cvwrite(vis, 'l2_er_' + restype + '_' + imtype + '.jpg', eval_dir + id + "/")

            error, vis = evaluation.ssim(gt, s)
            eval_vals[imtype][restype]["ssim"] = error
#        utils.cvwrite(vis, 'ssim_er_' + restype + '_' + imtype + '.jpg', eval_dir)

    with open(eval_dir + id + "/" + 'eval.json', 'w', encoding='utf-8') as json_file:
                json.dump(eval_vals, json_file, ensure_ascii=False, indent=4)

with open(eval_dir + 'full_eval.json', 'w', encoding='utf-8') as json_file:
            json.dump(full_eval, json_file, ensure_ascii=False, indent=4)
    '''


 
