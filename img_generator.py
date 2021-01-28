import cv2
import numpy as np

import optical_flow
import utils
from cubemapping import ExtendedCubeMap
from envmap import EnvironmentMap
from skimage.color import rgb2gray, gray2rgb

import matplotlib.pyplot as plt
import matplotlib
'''
#cubemap = cv2.cvtColor(cv2.imread("../../images/mapping_cube.jpg", 1), cv2.COLOR_BGR2RGB)
latlong = utils.load_img("../../images/04/square_L/L_max_vps_visualized_indices_clipped.jpg")
latlong = cv2.resize(latlong, (2000,1000))

#envmap = EnvironmentMap(cubemap, "cube")
envmap = EnvironmentMap(latlong, "latlong")
#envmap.convertTo("latlong")
#utils.cvwrite(envmap.data, "mapping_latlong.jpg")
envmap.convertTo("cube")
strip = utils.build_cube_strip_with_bottom(utils.split_cube(envmap.data))
utils.cvwrite(strip, "cubevis.jpg")

utils.cvwrite(envmap.data, "mapping_cube.jpg")
envmap.convertTo("sphere")
utils.cvwrite(envmap.data, "mapping_sphere.jpg")
envmap.convertTo("angular")
utils.cvwrite(envmap.data, "mapping_angular.jpg")
'''

'''
#calculate optical flow on a set of images and visualize
utils.build_params(p=0.5, l=10, w=12, i=20, path="../../text/360ImageSynthesisThesis/images/02/")
imgA = cv2.cvtColor(cv2.imread("../../text/360ImageSynthesisThesis/images/02/of_example1.jpg", 1), cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(cv2.imread("../../text/360ImageSynthesisThesis/images/02/of_example2.jpg", 1), cv2.COLOR_BGR2RGB)
gray = utils.load_img("../../text/360ImageSynthesisThesis/images/02/of_example1.jpg")
flow = optical_flow.farneback_of(imgA, imgB)
colorvis = optical_flow.visualize_flow(flow)
gray = rgb2gray(gray)
utils.print_type(gray)
gray = gray2rgb(gray)
utils.print_type(gray)
arrowvis = optical_flow.visualize_flow_arrows(gray, flow, 55)
utils.cvshow(arrowvis)
#utils.cvwrite(colorvis, "of_vis1.jpg")
utils.cvwrite(arrowvis, "of_vis2.jpg")
'''

'''
#calculate size of one face of the extended cube map
ecube = ExtendedCubeMap("../../data/captures/evaluation/checkersphere/images/0.jpg", 'latlong')
utils.cvshow(ecube.extended['front'])
print(ecube.extended["front"].shape)
'''

#scene = "checkersphere"
scene = "square_synth_room"
#scene = "oblong_room"
#scene = "sw"
dens = "6x6_dense"
id = "0"

#stripfunc = utils.build_sideways_cube
#stripfunc = utils.build_cube_strip
stripfunc = utils.build_cube_strip_with_bottom

try:
    gt_strip = utils.load_img(utils.OUT + scene + "_" + id + "_gt_strip.jpg")
    reg_strip = utils.load_img(utils.OUT + scene + "_" + id + "_reg_strip.jpg")
    reg_diff_strip = utils.load_img(utils.OUT + scene + "_" + id + "_reg_diff_strip.jpg")
    flow_strip = utils.load_img(utils.OUT + scene + "_" + id + "_flow_strip.jpg")
    flow_diff_strip = utils.load_img(utils.OUT + scene + "_" + id + "_flow_diff_strip.jpg")
except FileNotFoundError:

    #gt
    gt = utils.load_img('../../data/captures/thesis_selection/'+scene+'/'+dens+'/gt/gt_'+id+'.jpg')
    cube = EnvironmentMap(gt, "latlong").convertTo("cube").data
    gt_strip = stripfunc(utils.split_cube(cube))

    #reg result
    cube = utils.load_img('../../data/captures/thesis_selection/'+scene+'/'+dens+'/results/max_vps/'+id+'_max_vps_out_cube.jpg')
    reg_strip = stripfunc(utils.split_cube(cube))

    #reg diff
    cube = utils.load_img('../../data/captures/thesis_selection/'+scene+'/'+dens+'/results/max_vps/0eval/'+id+'_reg_l1.jpg')
    reg_diff_strip = stripfunc(utils.split_cube(cube))

    #flow result
    cube = utils.load_img('../../data/captures/thesis_selection/'+scene+'/'+dens+'/results/max_vps/'+id+'_max_vps_out_flow_cube.jpg')
    flow_strip = stripfunc(utils.split_cube(cube))

    #flow diff
    cube = utils.load_img('../../data/captures/thesis_selection/'+scene+'/'+dens+'/results/max_vps/0eval/'+id+'_flow_l1.jpg')
    flow_diff_strip = stripfunc(utils.split_cube(cube))

#    utils.cvwrite(gt_strip, scene + "_" + id + "_gt_strip.jpg")
#    utils.cvwrite(reg_strip, scene + "_" + id + "_reg_strip.jpg")
#    utils.cvwrite(reg_diff_strip, scene + "_" + id + "_reg_diff_strip.jpg")
#    utils.cvwrite(flow_strip, scene + "_" + id + "_flow_strip.jpg")
#    utils.cvwrite(flow_diff_strip, scene + "_" + id + "_flow_diff_strip.jpg")

fig, ax = plt.subplots(3,2)
ax[0,0].imshow(gt_strip)
ax[0,0].set_ylabel('ground truth')
ax[0,1].yaxis.set_label_position("right")
ax[1,0].imshow(reg_strip)
ax[1,0].set_ylabel('regular')
ax[2,0].imshow(flow_strip)
ax[2,0].set_ylabel('flow')
ax[1,1].imshow(reg_diff_strip)
ax[1,1].yaxis.set_label_position("right")
ax[1,1].set_ylabel('regular', rotation=270, labelpad=13)
ax[2,1].imshow(flow_diff_strip)
ax[2,1].yaxis.set_label_position("right")
ax[2,1].set_ylabel('flow', rotation=270, labelpad=13)
for x in ax.ravel():
    x.set_xticklabels([])
    x.set_yticklabels([])
    x.set_xticks([])
    x.set_yticks([])
    x.set_frame_on(False)

name = dens + '_' + id + '.jpg'

if stripfunc == utils.build_cube_strip:
#    ax[0,1].set_ylabel('L1 difference from ground truth\n(intensified)', rotation=0, labelpad=-90)
    ax[0,1].set_ylabel('L1 difference from ground truth', rotation=0, labelpad=-90, y=0.1)
    fig.subplots_adjust(wspace=0.04, hspace=-0.7)
    plt.savefig(utils.OUT + 'aux.jpg', bbox_inches='tight', dpi=300)#utils.DPI)
    fig = utils.load_img(utils.OUT + 'aux.jpg')
    fig = fig[255:,:,:]
    utils.cvwrite(fig, 'strip_' + name)

elif stripfunc == utils.build_sideways_cube:
#    ax[0,1].set_ylabel('L1 difference\nfrom ground truth\n(intensified)', rotation=0, labelpad=-115)
    ax[0,1].set_ylabel('L1 difference\nfrom ground truth', rotation=0, labelpad=-115, y=0.3)
    fig.subplots_adjust(wspace=-0.5, hspace=0.05)
    plt.savefig(utils.OUT + 'aux.jpg', bbox_inches='tight', dpi=400)#utils.DPI)
    fig = utils.load_img(utils.OUT + 'aux.jpg')
    fig = fig[:,:fig.shape[1]-255,:]
    utils.cvwrite(fig, 'cube_' + name)

elif stripfunc == utils.build_cube_strip_with_bottom:
#    ax[0,1].set_ylabel('L1 difference from ground truth\n(intensified)', rotation=0, labelpad=-90)
    ax[0,1].set_ylabel('L1 difference from ground truth', rotation=0, labelpad=-90, y=0.1)
    fig.subplots_adjust(wspace=-0.05, hspace=0.05)
    plt.savefig(utils.OUT + 'stripX_' + name, bbox_inches='tight', dpi=300)#utils.DPI)
#    fig = utils.load_img(utils.OUT + name)
#    fig = fig[255:,:,:]
#    utils.cvwrite(fig, 'strip_' + name)

#plt.show()
