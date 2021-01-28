import cv2
import numpy as np
import os
import string
import json

from envmap import EnvironmentMap
from skimage.metrics import structural_similarity

import utils
from preproc import parse_metadata
from cubemapping import ExtendedCubeMap

IDS = list(string.ascii_uppercase)[:25]

#plot colors from https://venngage.com/blog/color-blind-friendly-palette/#4
COLORS = ['#f5793a', '#a95aa1','#85c0f9', '#0f2080', '#fad9ad', '#ffffff']

BLENDTYPES = ["baseline", "regular", "flow"]
METRICTYPES = ["rgb_l1", "edge_l2", "gray_ssim_error ∈ [0,1]"]
USED_METRICTYPES = ["rgb_l1", "gray_ssim_error ∈ [0,1]"]
METRICTYPE_NAMES = {"rgb_l1":"L1 error", "edge_l2":"L2 error", "gray_ssim_error ∈ [0,1]":"SSIM error"}
                                
class ResultSet:
    '''
    '''
    def __init__(self, tlpath, blendtype, vps, ids, name=None, pos="", dens="", scene=""):
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
        self.dims = () 
        self.name = name
        self.dens = dens
        self.scene = scene
        self.pos = pos
        with open(self.tlpath + '../dims.txt', 'r') as f:
            data = f.readlines()
            dims0 = float(data[0].strip())
            dims1 = float(data[1].strip())
            self.dims = (dims0, dims1)

        self.radius = np.sqrt(2) * np.amax(self.dims)/2
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
        if self.name is None:
            with open(self.tlpath + 'name.txt', 'r') as f:
                name = f.readlines()[0].strip()
            return name# + "\n" + self.vps + " vps"
        else:
            return self.name

    def get_color(self):
        return COLORS[self.blendtype]

    def get_blendtype(self):
        return BLENDTYPES[self.blendtype]

    def get_gt_positions(self):
        gt_pos, _ = parse_metadata(self.tlpath + "gt_metadata.txt")
        return gt_pos * np.array([-1,1,1])

    def get_vps(self):
        pos, _ = parse_metadata(self.tlpath + "metadata.txt")
        return pos * np.array([-1,1,1])

def get_metric_name(metrictype):
    if metrictype in METRICTYPES:
        return METRICTYPE_NAMES[metrictype]
    else:
        raise(ValueError(metrictype + " is not a known metrictype"))

def get_color_by_blendtype(blendtype):
    if blendtype not in BLENDTYPES:
        raise TypeError("no type " + blendtype + " known")
    return COLORS[BLENDTYPES.index(blendtype)]

def get_blendtype_by_color(color):
    if color not in COLORS:
        raise TypeError("color " + color + " not in list of colors")
    return BLENDTYPES[COLORS.index(color)]

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
            id_dir = eval_dir# + id
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

            for pos, (restype, eval_types) in enumerate(eval_vals.items()):
                s = imgs["rgb"][restype]
                gt_l1 = imgs["rgb"]["gt"]

                #account for conversion problems
                dummy = None
#                if restype == "flow":
#                    dummy = ExtendedCubeMap(gt_l1, "cube").calc_clipped_cube()

#                if restype == "flow":
#                    gt_l1 = ExtendedCubeMap(gt_l1, "cube").calc_clipped_cube()

                error, vis = l1_error(gt_l1, s)
                if dummy is not None:
                    residual_error, _ = l1_error(gt_l1, dummy)
                    print("residual l1: ", residual_error)
                    l1_sum += residual_error
                    error -= residual_error
                eval_vals_np[i, pos*3 + 0] = error

#                print('rgb ', error)
#                utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_l1.jpg', eval_dir + id + "/")
                utils.cvwrite(vis, id + "_" + restype + '_l1.jpg', eval_dir)

                s = imgs["edges"][restype]
                gt_edges = imgs["edges"]["gt"]
                error, vis = l2_error(gt_edges, s)
                eval_vals_np[i, pos*3 + 1] = error
#                print('edge ', error)
#                utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_l2.jpg', eval_dir + id + "/")
#                utils.cvwrite(vis, id + "_" + restype + '_l2.jpg', eval_dir)

                s = imgs["gray"][restype]
                gt_ssim = imgs["gray"]["gt"]
#                if restype == "flow":
#                    gt_ssim = cv2.cvtColor(gt_l1.astype(np.float32), cv2.COLOR_RGB2GRAY)
                error, vis = ssim_error(gt_ssim, s)
                if dummy is not None:
                    dummygray = cv2.cvtColor(dummy.astype(np.float32), cv2.COLOR_RGB2GRAY)
                    residual_error, _ = ssim_error(gt_ssim, dummygray)
                    ssim_sum += residual_error
                    error -= residual_error
                    print("residual ssim: ", residual_error)
                    print()
                eval_vals_np[i, pos*3 + 2] = error
#        utils.cvwrite(vis, id + "_" + paramstring + "_" + restype + '_ssim.jpg', eval_dir + id + "/")

        with open(out_dir + '0eval/full_eval.npy', 'wb') as f:
            np.save(f, eval_vals_np)

#        print("l1 error sum: ", l1_sum)
#        print("l1 average extra error: ", l1_sum/len(ids))
#        print("number of ids: ", len(ids))
#        print("ssim error sum: ", ssim_sum)
#        print("ssim average extra error: ", ssim_sum/len(ids))

        #extract order of results by viewpoint and metric
        sortedIDs = {BLENDTYPES[0]: {}, BLENDTYPES[1]: {}, BLENDTYPES[2]: {}}
        for blendnum in range(len(BLENDTYPES)):
            for metricnum in range(len(METRICTYPES)):
                sorted_column = np.argsort(eval_vals_np[:, blendnum*3+metricnum], axis=0)
                asciiIDs = []
                for idnum in sorted_column:
                    if sorted_column.shape[0] > 25:
                        asciiIDs.append(str(idnum) + ": " + str(eval_vals_np[:, blendnum*3+metricnum][idnum]))
                    else:
                        asciiIDs.append(string.ascii_uppercase[idnum] + ": " + str(eval_vals_np[:, blendnum*3+metricnum][idnum]))

                sortedIDs[BLENDTYPES[blendnum]][METRICTYPES[metricnum]] = asciiIDs
                with open(out_dir + '0eval/ID_order.json', 'w') as f:
                    json.dump(sortedIDs, f, indent=4)

        return eval_vals_np

def l1_error(gt, s):
    '''
    gt: ground truth
    s: synthesized image
    calculates the absolute error pixel-wise of two images in RGB color space 
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
    facewidth = float(vis.shape[0]) / 4.0
    error /= (facewidth * facewidth) * 6
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
    facewidth = float(vis.shape[0]) / 4.0
    error /= (facewidth * facewidth) * 6
    return (error, vis)

def l1_error_regular(gt, s):
    vis = np.sum(np.abs(gt - s), -1)
    error = np.sum(vis)
    #calculate mean
    error /= vis.shape[0] * vis.shape[1]
    return (error, vis)

def ssim_error(gt, s):
    '''
    calculates the structural similarity which yields a value between -1 and 1 (1 being identical)
    converts the ssim value to an error value in the interval [0,1], 0 being no error --> identical
    '''
#    if not ( utils.is_cubemap(gt) and utils.is_cubemap(s) ):
#        raise TypeError('The input must be in cubemap representation. Pass through prep_image first')
    if len(gt.shape) > 2:
        mssim, vis = structural_similarity(gt, s, multichannel=True, full=True)
    else:
        mssim, vis = structural_similarity(gt, s, full=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    error = (-mssim + 1)/2
    return(error, vis)

def prep_image(img):
    if not utils.is_cubemap(img):
        img = EnvironmentMap(img, "latlong").convertTo("cube").data.astype(np.float32)
    return img

