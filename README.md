# 360imageSynthesis
This is the implementation of the **pixel-based 2-degree-of-freedom synthesis of 360° viewpoints using flow-based interpolation** introduced in my master's thesis of the same name. The thesis can be found at: http://www.nm.ifi.lmu.de/pub/Diplomarbeiten/klet21/

## Requirements
Python 3.7

Libraries:

    https://github.com/soravux/skylibs

    https://github.com/soravux/rotlib

For other dependencies, see the venv-generated requirements.txt. You can install them handily using
```BASH
pip install -r requirements.txt
```


## Input Data
Before synthesizing viewpoints in a scene, the scene needs to be captured using a 360° camera. All images need to be on a plane, so using a tripod is recommended. The metadata can be extracted by using a structure-from-motion library, like OpenSfM (https://www.opensfm.org/). An example scene is stored in testdata/VirtualRoom_CaptureSet. Any scene needs to be stored in a directory with the following structure:

scenedirectory/

|---metadata.txt

|---images/

    |--0.jpg
    
    |--1.jpg
    
    |...

metadata.txt: a file containing the metadata of the captures, see preproc.parse_metadata() and the example metadata.txt file for formatting details

images/: a directory containing the image data in latlong format, numbered from 0 to N with no leading zeros, in the same order as the metadata


## Execution
For free 2DoF synthesis within a scene, use main_2DoF.py, replacing the path of the CaptureSet with your own.
```BASH
python3 main_2DoF.py
```
For 1DoF synthesis between a pair of images, use main_1DoF.py, replacing the path of the two images with your own.
