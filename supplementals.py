import cv2
import numpy as np
import optical_flow, utils
import matplotlib.pyplot as plt
import os
import time

def render_of(indices, blenderfile, location):
    '''
    OpticalFlow at 1001 is A-B and OpticalFlow at 1002 is B-A
    build flowcube
    return A-B and B-A
    '''
    print('render of')
    path = '../../data/tmp/'
    if location is not None:
        code = os.system('blender '+ location + blenderfile + ' --background --python ./blender_scripts/blender_optical_flow.py ' + ' -- ' + str(indices[0]) + ' ' + str(indices[1]))
    else:
        code = os.system('ssh -l ubuntu 10.195.1.224 blender /home/ubuntu/blenderfiles/'+ blenderfile + ' --background --python ./blenderfiles/blender_optical_flow.py ' + ' -- ' + str(indices[0]) + ' ' + str(indices[1]))
        print("remote call exit code: ", code)

        lock = ""
        while lock != str(indices[0]) + "-" + str(indices[1]):
            time.sleep(10)
            lock = os.popen('ssh -l ubuntu 10.195.1.224 cat optical_flow/lock.txt').read().strip()

        #scp files to path
        code = os.system('scp ubuntu@10.195.1.224:~/optical_flow/* ' + path)

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
