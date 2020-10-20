import cv2
import numpy as np
import optical_flow, utils
import matplotlib.pyplot as plt
import os

def blender_exr2flow(path):
    """
    Adapted from http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    """
    exr = cv2.imread(path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)
    flow = np.dstack((exr[:,:,0], -exr[:,:,1]))
#    plt.imshow(exr[:,:,0])
#    plt.show()
#    plt.imshow(exr[:,:,1])
#    plt.show()
#    plt.imshow(exr[:,:,2])
#    plt.show()
    return flow

def render_of(indices, blenderfile):
    stream = os.system('blender '+ blenderfile + ' --background --python render_of.py ' + ' -- ' + str(indices[0]) + ' ' + str(indices[1]))
    '''
    OpticalFlow at 1001 is B-A and OpticalFlow at 1002 is A-B SEEMS OPPOSITE
    build flowcube
    return A-B and B-A
    '''
    path = '../../data/tmp/'
    ofs = {}
    for ofname in ["OpticalFlow1002", "OpticalFlow1001"]:
        faces = {}
        for face in utils.FACES:
            facepath = path + ofname + "_" + face + ".exr"
            exr = cv2.imread(facepath, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)
            faces[face] = -np.dstack((exr[:,:,0], -exr[:,:,1]))
        flow = utils.build_cube(faces)
#        utils.cvshow(optical_flow.visualize_flow(flow), ofname)
        ofs[ofname] = flow
    flowAB = ofs["OpticalFlow1001"]
    flowBA = ofs["OpticalFlow1002"]

    return (flowAB, flowBA)

#def build_flowcube(path):
#    files = sorted(listdir(path + "/back"))
#    for f in files:
#        name, extension = os.path.splitext(f)
#        if extension == ".exr":
#            print(f)
#            faces = {}
#            for face in FACES:
#                facepath = path + face + "/" + f
#                print(facepath)
#                exr = cv2.imread(facepath, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
#                exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB)
#                faces[face] = -np.dstack((exr[:,:,0], -exr[:,:,1]))
#            flow = utils.build_cube(faces)
#            with open(path + '/' + name + '.npy', 'wb') as f:
#                    np.save(f, flow)
