import bpy
import sys
import numpy as np

pps = int(sys.argv[-2]) #points per side
width = int(sys.argv[-1]) #width of the scene (square)

dist = width/pps

points = np.zeros((pps, pps, 2))
for x in range(pps):
    for y in range(pps):
        points[y,x] = np.array([x*dist + dist/2, y*dist + dist/2])

points = points - np.array([width/2, width/2])
points = points.reshape((points.shape[0] * points.shape[1], points.shape[2]))
'''
#Y AXIS IS FLIPPED FOR BLENDER!!
points = np.array([  [-1, 1], [-1, 0.5], [-1, 0], [-1, -0.5], [-1, -1],
            [-0.5, 1], [-0.5, 0.5], [-0.5, 0], [-0.5, -0.5], [-0.5, -1],
            [0, 1], [0, 0.5], [0, 0], [0, -0.5], [0, -1],
            [0.5, 1], [0.5, 0.5], [0.5, -0], [0.5, -0.5], [0.5, -1],
            [1, 1], [1, 0.5], [1, 0], [1, -0.5], [1, -1],
        ]) #gt points
'''

with open(bpy.path.abspath("//") + '/metadata.txt', 'w', encoding='utf-8') as f:
    f.write('0,0,0\n1,-1,1\n')

cam = bpy.context.scene.objects["Camera"]
for i in range(pps * pps):
    #set the keyframe at the correct position
    bpy.context.scene.frame_set(i)
    cam.location = (points[i,0], points[i,1], 0)
    cam.keyframe_insert(data_path="location", frame=i)
    bpy.context.scene.update()

    #save as metadata
    with open(bpy.path.abspath("//") + '/metadata.txt', 'a', encoding='utf-8') as f:
        f.write('0.0, 0.0, 0.0, 1.0, ')
        f.write(str(points[i,0]) + ', ' + str(points[i,1]) + ', 0\n')

bpy.ops.wm.save_mainfile()



