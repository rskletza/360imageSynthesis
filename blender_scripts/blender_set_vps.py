import bpy
import sys
import numpy as np

'''
blender <blenderfile> --background --python blender_set_vps.py -- <points per side> <width of the scene in meters>
'''

type = sys.argv[-3] #ground truth points or regular ([gt | vps])
pps = int(sys.argv[-2]) #points per side
width = float(sys.argv[-1]) #width of the scene (square)


points = np.zeros((pps, pps, 2))
if type == "vps":
    dist = width/pps
    docstring = "/metadata.txt"
    for x in range(pps):
        for y in range(pps):
            points[y,x] = np.array([x*dist + dist/2, y*dist + dist/2])


elif type == "gt":
    dist = width/6
    docstring = "/gt_metadata.txt"
    for x in range(5):
        for y in range(5):
            points[y,x] = np.array([x*dist + dist, y*dist + dist])

#center points around 0
points = points - np.array([width/2, width/2])
#flatten
points = points.reshape((points.shape[0] * points.shape[1], points.shape[2]))
'''
points = np.array([  [-1, 1], [-1, 0.5], [-1, 0], [-1, -0.5], [-1, -1],
            [-0.5, 1], [-0.5, 0.5], [-0.5, 0], [-0.5, -0.5], [-0.5, -1],
            [0, 1], [0, 0.5], [0, 0], [0, -0.5], [0, -1],
            [0.5, 1], [0.5, 0.5], [0.5, -0], [0.5, -0.5], [0.5, -1],
            [1, 1], [1, 0.5], [1, 0], [1, -0.5], [1, -1],
        ]) #gt points
'''

print("writing " + type + " metadata")
with open(bpy.path.abspath("//") + docstring, 'w', encoding='utf-8') as f:
    f.write('0,0,0\n1,-1,1\n')

cam = bpy.context.scene.objects["Camera"]
for i in range(pps * pps):
    #set the keyframe at the correct position
    bpy.context.scene.frame_set(i)
    cam.location = (points[i,0], points[i,1], 0)
    cam.keyframe_insert(data_path="location", frame=i)
    bpy.context.scene.update()

    #save as metadata
    with open(bpy.path.abspath("//") + docstring, 'a', encoding='utf-8') as f:
        f.write('0.0, 0.0, 0.0, 1.0, ')
        f.write(str(points[i,0]) + ', ' + str(points[i,1]) + ', 0\n')

bpy.ops.wm.save_mainfile()
