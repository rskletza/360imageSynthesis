from evaluation import ResultSet
from evaluation import IDS

'''
#comparing min vs max viewpoints
tlpath1 = "../../data/captures/thesis_selection/checkersphere/6x6/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
tlpath3 = "../../data/captures/thesis_selection/oblong_room/6x6/"
out_type = "regular" # "regular" or "flow"
name = "Comparing min vs max viewpoints for " + out_type + " blending"
res_sets = [
        [
#            ResultSet(tlpath1, "baseline", "max", IDS),
            ResultSet(tlpath1, out_type, "min", IDS),
            ResultSet(tlpath1, out_type, "max", IDS)
        ]
        ,
        [
#            ResultSet(tlpath2, "baseline", "max", IDS),
            ResultSet(tlpath2, out_type, "min", IDS),
            ResultSet(tlpath2, out_type, "max", IDS)
        ]
        ,
        [
#            ResultSet(tlpath3, "baseline", "max", IDS),
            ResultSet(tlpath3, out_type, "min", IDS),
            ResultSet(tlpath3, out_type, "max", IDS)
        ]
    ]

'''
#comparing viewpoint densities
tlpath1 = "../../data/captures/thesis_selection/square_synth_room/2x2/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
tlpath3 = "../../data/captures/thesis_selection/square_synth_room/12x12/"
name = "Different viewpoint densities in square_room"
res_sets = [
        [
            ResultSet(tlpath1, "baseline", "max", IDS),
            ResultSet(tlpath2, "baseline", "max", IDS),
            ResultSet(tlpath3, "baseline", "max", IDS)
        ] ,
        [
            ResultSet(tlpath1, "regular", "max", IDS),
            ResultSet(tlpath2, "regular", "max", IDS),
            ResultSet(tlpath3, "regular", "max", IDS)
        ],
        [
            ResultSet(tlpath1, "flow", "max", IDS),
            ResultSet(tlpath2, "flow", "max", IDS),
            ResultSet(tlpath3, "flow", "max", IDS)
        ],
    ]
'''

#comparing model-scene difference
tlpath1 = "../../data/captures/thesis_selection/checkersphere/6x6/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
tlpath3 = "../../data/captures/thesis_selection/oblong_room/6x6/"
tlpath4 = "../../data/captures/thesis_selection/oblong_room_v2/6x6/"
name = "Comparing flow and regular blending in different scenes"
res_sets = [
        [
#            ResultSet(tlpath1, "baseline", "max", IDS),
            ResultSet(tlpath1, "regular", "max", IDS),
            ResultSet(tlpath1, "flow", "max", IDS)
        ],
        [
            ResultSet(tlpath2, "baseline", "max", IDS),
            ResultSet(tlpath2, "regular", "max", IDS),
            ResultSet(tlpath2, "flow", "max", IDS)
        ]
        ,
        [
            ResultSet(tlpath3, "baseline", "max", IDS),
            ResultSet(tlpath3, "regular", "max", IDS),
            ResultSet(tlpath3, "flow", "max", IDS)
        ],
        [
            ResultSet(tlpath4, "baseline", "max", IDS),
            ResultSet(tlpath4, "regular", "max", IDS),
            ResultSet(tlpath4, "flow", "max", IDS)
        ],
    ]
'''

'''
tlpath = "../../data/captures/thesis_selection/square_synth_room/6x6/"
pos_name = "Detailed comparison of flow and regular blending in square_room6x6"
pos_res_sets = [
    ResultSet(tlpath, "regular", "max", IDS),
    ResultSet(tlpath, "flow", "max", IDS)
        ]
'''

#comparing viewpoint densities
tlpath1 = "../../data/captures/thesis_selection/square_synth_room/2x2/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
tlpath3 = "../../data/captures/thesis_selection/square_synth_room/12x12/"
name = "Different viewpoint densities in square_room"
pos_res_sets = [
    ResultSet(tlpath1, "regular", "max", IDS),
#    ResultSet(tlpath2, "regular", "max", IDS),
#    ResultSet(tlpath3, "regular", "max", IDS),
    ResultSet(tlpath1, "flow", "max", IDS),
#    ResultSet(tlpath2, "flow", "max", IDS),
#    ResultSet(tlpath3, "flow", "max", IDS),
        ]
