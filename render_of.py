import bpy
import sys
import copy

'''
command line arguments are
1: first index
2: second index

copy keyframe of Camera at indices[0] to position 1000 and 1002
copy keyframe of Camera at indices[1] to position 1001
TODO better: copy the keyframes and then delete all (this does not limit the number of viewpoints to 1000)
render keyframe 1001 and 1002
'''

indexA = int(sys.argv[-2])
indexB = int(sys.argv[-1])
print("getting blender optical flow between viewpoints {0:d} and {1:d}".format(indexA, indexB))

cam = bpy.context.scene.objects["Camera"]

bpy.context.scene.frame_set(indexA)
locationA = copy.deepcopy(cam.location)
#TODO if location.copy works as well, dont need the import
bpy.context.scene.frame_set(1000)
cam.location = locationA
bpy.context.scene.update()
cam.keyframe_insert(data_path="location", frame=1000)
cam.keyframe_insert(data_path="location", frame=1002)

bpy.context.scene.frame_set(indexB)
locationB = cam.location.copy()#copy.deepcopy(cam.location)
bpy.context.scene.frame_set(1001)
cam.location = locationB
bpy.context.scene.update()
cam.keyframe_insert(data_path="location", frame=1001)

bpy.data.scenes["Scene"].frame_start = 1001
bpy.data.scenes["Scene"].frame_end = 1002
bpy.ops.render.render(animation=True)

print(locationA)
print(locationB)
bpy.context.scene.frame_set(1000)
print(cam.location)
bpy.context.scene.frame_set(1001)
print(cam.location)
bpy.context.scene.frame_set(1002)
print(cam.location)





