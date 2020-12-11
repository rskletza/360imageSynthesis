import cv2
import numpy as np
import os
import string
import json

from envmap import EnvironmentMap
from skimage.metrics import structural_similarity

import utils

#plot colors from https://venngage.com/blog/color-blind-friendly-palette/#4
COLORS = ['#f5793a', '#a95aa1','#85c0f9', '#0f2080', '#fad9ad', '#ffffff']

BLENDTYPES = ["baseline", "regular", "flow"]
METRICTYPES = ["rgb_l1", "edge_l2", "gray_ssim_error ∈ [0,1]"]
                                
class ResultSet:
    '''
    '''
    def __init__(self, tlpath, blendtype, vps, ids):
        '''
        tlpath: top level path (directory containing gt folder and results folder)
        blendtype: "flow" or "regular" or "baseline"
        vps: "max" or "min"
        '''
        self.tlpath = tlpath
        self.gt = tlpath + "gt/"
        self.ids = ids
        self.respath = tlpath + "results/" + vps + "_vps/"
        self.vps = vps
        if blendtype in BLENDTYPES:
            #(0: baseline, 1: regular, 2: flow)
            self.blendtype = BLENDTYPES.index(blendtype)
        else:
            raise(ValueError(blendtype + " is not a known blendtype"))

        self.metrics = self.get_metrics()
        #first column is l1 rgb, second is ssim, third is l2

    def get_metrics(self):
        try:
            with open(self.respath + '0eval/full_eval.npy', 'rb') as f:
                all_eval_vals = np.load(f)

        except(FileNotFoundError):
            all_eval_vals = calc_metrics(self.respath, self.gt, self.ids)

        #get only the values for the blendtype 
        eval_vals = all_eval_vals[:,self.blendtype*3:self.blendtype*3+3]
        return eval_vals

    def get_metrics_by_type(self, metrictype):
        if metrictype in METRICTYPES:
            mpos = METRICTYPES.index(metrictype)
        else:
            raise(ValueError(metrictype + " is not a known metrictype"))
        metrics = self.get_metrics()[:,mpos]
#        print(mpos)
#        print(metrictype)
#        print("get metrics by type: ", METRICTYPES[mpos], self.vps)
#        print(metrics)
        return metrics

    def get_name(self):
        with open(self.tlpath + 'name.txt', 'r') as f:
            name = f.readlines()[0].strip()
        return name + "\n" + BLENDTYPES[self.blendtype] + ", " + self.vps + " vps"

    def get_color(self):
        return COLORS[self.blendtype]

    '''
    def boxplot(self, ax, index, metrictype):
        if metrictype in METRICTYPES:
            mpos = METRICTYPES.index(metrictype)
        else:
            raise(ValueError(metrictype + " is not a known metrictype"))

        ax.boxplot(self.metrics[])
    '''



def calc_metrics(out_dir, gt_dir, ids):
    '''
    out_dir: the directory containing the results
    '''
    try:
        with open(out_dir + '0eval/full_eval.npy', 'rb') as f:
            eval_vals_np = np.load(f)
        return eval_vals_np

    except(FileNotFoundError):
        eval_dir = out_dir + "0eval/" #the 0 is for placement in the folder
        if not os.path.exists(eval_dir):
            try:
                os.mkdir(eval_dir)
            except OSError as exc: # guard agains race condition
                if exc.ernno != errno.EEXIST:
                        raise

        eval_vals_np = np.full((len(ids), 9), np.Inf)
        vp_type = out_dir.split("/")[-2] #max_vps or min_vps

        for i, id in enumerate(ids):
            #create a folder with this id name in the eval_dir
            id_dir = eval_dir + id
            if not os.path.exists(id_dir):
                try:
                    os.mkdir(id_dir)
                except OSError as exc: # guard agains race condition
                    if exc.ernno != errno.EEXIST:
                            raise

            #for each output image ..
            img_reg = prep_image(utils.load_img(out_dir + id + "_" + vp_type + "_out_cube.jpg")) #the 'regular' output image with deviation-angle-based blending

            img_flow = prep_image(utils.load_img(out_dir + id + "_" + vp_type + "_out_flow_cube.jpg")) #the output image with flow-based blending

            img_baseline = prep_image(utils.load_img(out_dir + id + "_" + vp_type + "_baseline_cube.jpg"))  #the baseline image

            gt = prep_image(utils.load_img(gt_dir + "gt_" + id + ".jpg"))

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
            #matrix contains first all baseline metrics, then all regular metrics then all flow metrics --> 3*3 = 9 values per index

            #TODO is the faulty conversion from cube to latlong actually an issue in error eval because evaluation is done in cubemap anyway? yes because some of the artefacts are transferred elsewhere in the image

            for pos, (restype, eval_types) in enumerate(eval_vals.items()):
                s = imgs["rgb"][restype]
                gt = imgs["rgb"]["gt"]
                error, vis = l1_error(gt, s)
                eval_vals_np[i, pos*3 + 0] = error
#                print('rgb ', error)
#                utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_l1.jpg', eval_dir + id + "/")
                utils.cvwrite(vis, id + "_" + restype + '_l1.jpg', eval_dir + id + "/")

                s = imgs["edges"][restype]
                gt = imgs["edges"]["gt"]
                error, vis = l2_error(gt, s)
                eval_vals_np[i, pos*3 + 1] = error
#                print('edge ', error)
#                utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_l2.jpg', eval_dir + id + "/")
                utils.cvwrite(vis, id + "_" + restype + '_l2.jpg', eval_dir + id + "/")

                s = imgs["gray"][restype]
                gt = imgs["gray"]["gt"]
                error, vis = ssim_error(gt, s)
                eval_vals_np[i, pos*3 + 2] = error
#        utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_ssim.jpg', eval_dir + id + "/")

        with open(out_dir + '0eval/full_eval.npy', 'wb') as f:
            np.save(f, eval_vals_np)

        #extract order of results by viewpoint and metric
        sortedIDs = {BLENDTYPES[0]: {}, BLENDTYPES[1]: {}, BLENDTYPES[2]: {}}
        for blendnum in range(len(BLENDTYPES)):
            for metricnum in range(len(METRICTYPES)):
                sorted_column = np.argsort(all_eval_vals[:, blendnum*3+metricnum], axis=0)
                asciiIDs = []
                for idnum in sorted_column:
                    asciiIDs.append(string.ascii_uppercase[idnum])

                sortedIDs[BLENDTYPES[blendnum]][METRICTYPES[metricnum]] = asciiIDs
                with open(out_dir + '0eval/ID_order.json', 'w') as f:
                    json.dump(sortedIDs, f, indent=4)

        return eval_vals_np

def boxplot(eval_vals, out_type, vps):
    '''
    out_type: flow or res
    vps: min or max
    '''

def plot_allvps(eval_vals_np, saveas=(None, "")):
    '''
    DONT USE, TOO MANY CHANGES
    eval_vals_np: 
    saveas: (path without filename, parameter string for filename)
    '''
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

    ax[1].plot(x, eval_vals_np[:,1], color=color1, linestyle='--') #baseline edge
    ax[1].plot(x, eval_vals_np[:,4], color=color2, linestyle='-') #reg edge
    ax[1].plot(x, eval_vals_np[:,7], color=color3, linestyle='-.') #flow edge
    ax[1].set_title('Inverted SSIM measure on grayscale images (ssim)')
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

    if saveas is not None:
        plt.savefig(saveas + 'eval_graph_' + paramstring + '.png', bbox_inches='tight')
    plt.show()
    plt.clf()

def l1_error(gt, s):
    '''
    gt: ground truth
    s: synthesized image
    calculates the absolute error pixel-wise of two images in CIELAB color space 
    '''
    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')

    if len(s.shape) > 2:
        vis = np.sum(np.abs(gt - s), -1)
    else:
        vis = np.abs(gt - s)
    #sum up complete error (since the black areas are 0, they have no impact)
    error = np.sum(vis)
    #calculate mean of the faces (not of the whole image, as the black areas would influence the result (because they have 0 error)
    error /= (vis.shape[0] / 4) * 6
    return (error, vis)

def l2_error(gt, s):
    '''
    calculates the squared error pixel-wise of two images
    '''
    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')

    if len(s.shape) > 2:
        vis = np.sum(np.power(gt - s, 2), -1)
    else:
        vis = np.power(gt - s, 2)

    #sum up complete error (since the black areas are 0, they have no impact)
    error = np.sum(vis)
    #calculate mean of the faces (not of the whole image, as the black areas would influence the result (because they have 0 error)
    error /= (vis.shape[0] / 4) * 6
    return (error, vis)

def ssim_error(gt, s):
    '''
    calculates the structural similarity which yields a value between -1 and 1 (1 being identical)
    converts the ssim value to an error value in the interval [0,1], 0 being no error --> identical
    '''
    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')
    if len(gt.shape) > 2:
        mssim, vis = structural_similarity(gt, s, multichannel=True, full=True)
    else:
        mssim, vis = structural_similarity(gt, s, full=True)
    error = (-mssim + 1)/2
    return(error, vis)

def prep_image(img):
    if not utils.is_cubemap(img):
        img = EnvironmentMap(img, "latlong").convertTo("cube").data.astype(np.float32)
    return img

