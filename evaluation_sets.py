import string
from evaluation import ResultSet

ids = list(string.ascii_uppercase)[:25]

'''
#comparing min vs max viewpoints
tlpath1 = "../../data/captures/thesis_selection/checkersphere/6x6/checkersphere_6vps_500x1000/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
out_type = "regular" # "regular" or "flow"
name = "Comparing min vs max viewpoints for " + out_type + " blending"
res_sets = [
        [
#            ResultSet(tlpath1, "baseline", "max", ids),
            ResultSet(tlpath1, out_type, "min", ids),
            ResultSet(tlpath1, out_type, "max", ids)
        ]
        ,
        [
#            ResultSet(tlpath2, "baseline", "max", ids),
            ResultSet(tlpath2, out_type, "min", ids),
            ResultSet(tlpath2, out_type, "max", ids)
        ]
    ]

'''
'''
#comparing viewpoint densities
tlpath1 = "../../data/captures/thesis_selection/square_synth_room/2x2/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
tlpath3 = "../../data/captures/thesis_selection/square_synth_room/12x12/"
name = "Different viewpoint densities in square_room"
res_sets = [
#        [
#            ResultSet(tlpath1, "baseline", "max", ids),
#            ResultSet(tlpath1, "regular", "max", ids),
#            ResultSet(tlpath1, "flow", "max", ids)
#        ] ,
        [
            ResultSet(tlpath2, "baseline", "max", ids),
            ResultSet(tlpath2, "regular", "max", ids),
            ResultSet(tlpath2, "flow", "max", ids)
        ],
        [
            ResultSet(tlpath3, "baseline", "max", ids),
            ResultSet(tlpath3, "regular", "max", ids),
            ResultSet(tlpath3, "flow", "max", ids)
        ],
    ]

#comparing model-scene difference
tlpath1 = "../../data/captures/thesis_selection/checkersphere/6x6/checkersphere_6vps_500x1000/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6/"
name = "Comparing flow and regular blending in different scenes"
res_sets = [
        [
            ResultSet(tlpath1, "baseline", "max", ids),
            ResultSet(tlpath1, "regular", "max", ids),
            ResultSet(tlpath1, "flow", "max", ids)
        ],
        [
            ResultSet(tlpath2, "baseline", "max", ids),
            ResultSet(tlpath2, "regular", "max", ids),
            ResultSet(tlpath2, "flow", "max", ids)
        ]
#        ,
#        [
#            ResultSet(tlpath3, "baseline", "max", ids),
#            ResultSet(tlpath3, "regular", "max", ids),
#            ResultSet(tlpath3, "flow", "max", ids)
#        ],
    ]
'''

