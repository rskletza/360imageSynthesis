import bpy
import os
import sys

pps = int(sys.argv[-1]) #points per side

bpy.data.scenes["Scene"].render.filepath = os.getenv("HOME") + "render/"

bpy.data.scenes["Scene"].frame_start = 0
bpy.data.scenes["Scene"].frame_end = pps * pps - 1
bpy.ops.render.render(animation=True)
