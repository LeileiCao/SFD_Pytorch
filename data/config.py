# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("/data1/caoleilei")
# note: if you used our download scripts, this should be right
FACEroot = os.path.join(home,'data/widerface')

#RFB CONFIGS
FACE = {
    'feature_maps' : [160, 80, 40, 20, 10, 5],

    'min_dim' : 640,

    'steps' : [4, 8, 16, 32, 64, 128],

    'min_sizes' : [16, 32, 64, 128, 256, 512],

    'variance' : [0.1, 0.2],

    'clip' : True,
}
