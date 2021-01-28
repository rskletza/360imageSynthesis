from evaluation import ResultSet
from evaluation import IDS
'''
#comparing model-scene difference
tlpath1 = "../../data/captures/thesis_selection/checkersphere/6x6/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6_offset/"
tlpath3 = "../../data/captures/thesis_selection/oblong_room/6x6_offset/"
tlpath4 = "../../data/captures/thesis_selection/oblong_room_v2/6x6_offset/"
name = "all_boxplot"
res_sets = [
        [
#            ResultSet(tlpath1, "baseline", "max", IDS),
            ResultSet(tlpath1, "regular", "max", IDS, name="checker-\nsphere"),
            ResultSet(tlpath1, "flow", "max", IDS, name="checker-\nsphere")
        ],
       [
            ResultSet(tlpath2, "baseline", "max", IDS, name="square\nroom"),
            ResultSet(tlpath2, "regular", "max", IDS, name="square\nroom"),
            ResultSet(tlpath2, "flow", "max", IDS, name="square\nroom")
       ]
        ,
        [
            ResultSet(tlpath3, "baseline", "max", IDS, name="oblong\nroom"),
            ResultSet(tlpath3, "regular", "max", IDS, name="oblong\nroom"),
            ResultSet(tlpath3, "flow", "max", IDS, name="oblong\nroom")
        ],

#        [
#            ResultSet(tlpath1, "regular", "max", IDS, name="checker-\nsphere"),
#            ResultSet(tlpath2, "regular", "max", IDS, name="square\nroom"),
#            ResultSet(tlpath3, "regular", "max", IDS, name="oblong\nroom"),
#        ],
#       [
#            ResultSet(tlpath1, "flow", "max", IDS, name="checker-\nsphere"),
#            ResultSet(tlpath2, "flow", "max", IDS, name="square\nroom"),
#            ResultSet(tlpath3, "flow", "max", IDS, name="oblong\nroom")
#       ]
    ]

'''
#comparing viewpoint densities
tlpath1 = "../../data/captures/thesis_selection/square_synth_room/2x2v2_offset_new/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6_offset_new/"
tlpath3 = "../../data/captures/thesis_selection/square_synth_room/12x12_offset_new/"
name = "dens_boxplot"
res_sets = [
#        [
#            ResultSet(tlpath1, "baseline", "max", IDS, name="2x2"),
#            ResultSet(tlpath2, "baseline", "max", IDS, name="6x6"),
#            ResultSet(tlpath3, "baseline", "max", IDS, name="12x12")
#        ] ,
#        [
#            ResultSet(tlpath1, "regular", "max", IDS, name="2x2"),
#            ResultSet(tlpath2, "regular", "max", IDS, name="6x6"),
#            ResultSet(tlpath3, "regular", "max", IDS, name="12x12")
#        ],
#        [
#            ResultSet(tlpath1, "flow", "max", IDS, name="2x2"),
#            ResultSet(tlpath2, "flow", "max", IDS, name="6x6"),
#            ResultSet(tlpath3, "flow", "max", IDS, name="12x12")
#        ],

        [
            ResultSet(tlpath1, "baseline", "max", IDS, name="2x2"),
            ResultSet(tlpath1, "regular", "max", IDS, name="2x2"),
            ResultSet(tlpath1, "flow", "max", IDS, name="2x2"),
        ],
        [
            ResultSet(tlpath2, "baseline", "max", IDS, name="6x6"),
            ResultSet(tlpath2, "regular", "max", IDS, name="6x6"),
            ResultSet(tlpath2, "flow", "max", IDS, name="6x6"),
        ],
        [
            ResultSet(tlpath3, "baseline", "max", IDS, name="12x12"),
            ResultSet(tlpath3, "regular", "max", IDS, name="12x12"),
            ResultSet(tlpath3, "flow", "max", IDS, name="12x12"),
        ]
    ]

'''
#comparing dense positions
tlpath = "../../data/captures/thesis_selection/square_synth_room/6x6_dense/"
name = "dense"
IDS = list(map(str, list(range(25*25))))
res_sets = [
        [
#            ResultSet(tlpath, "baseline", "max", IDS, name="square\nroom"),
            ResultSet(tlpath, "regular", "max", IDS, name="square\nroom"),
            ResultSet(tlpath, "flow", "max", IDS, name="square\nroom"),
            ]
    ]

'''
'''
#testing error correction
tlpath1 = "../../data/captures/thesis_selection/oblong_room_v2/6x6_offset/"
tlpath2 = "../../data/captures/thesis_selection/oblong_room_v2/6x6_offset_corr_error_minus/"
tlpath3 = "../../data/captures/thesis_selection/oblong_room_v2/6x6_offset_corr_error_gt/"
name = "error_corr"
res_sets = [

        [
            ResultSet(tlpath1, "baseline", "max", IDS, name="normal oblong"),
            ResultSet(tlpath1, "regular", "max", IDS, name="normal_oblong"),
        ],
        [
            ResultSet(tlpath1, "flow", "max", IDS, name="normal_oblong"),
            ResultSet(tlpath3, "flow", "max", IDS, name="corr. oblong gt"),
            ResultSet(tlpath2, "flow", "max", IDS, name="corr. oblong minus"),
        ],
    ]


#real scene: comparing baseline, regular and flow, min and max
#tlpath = "../../data/captures/thesis_selection/herrsching/5x5_close/"
tlpath = "../../data/captures/thesis_selection/sw/5x5_random/"
name = "Real Scene Results"
res_sets = [
        [
            ResultSet(tlpath, "baseline", "max", IDS[:13], name="naive"),
            ResultSet(tlpath, "regular", "max", IDS[:13], name="regular"),
            ResultSet(tlpath, "flow", "max", IDS[:13], name="flow"),
        ],
    ]
'''

############################# pos sets ####################################

'''
#comparing proxy-scene difference
tlpath1 = "../../data/captures/thesis_selection/checkersphere/6x6/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6_offset/"
tlpath3 = "../../data/captures/thesis_selection/oblong_room/6x6_offset/"
pos_res_sets = [
#    ResultSet(tlpath1, "regular", "max", IDS, name="checkersphere,\nregular blending", pos="", scene="sphere"),
#    ResultSet(tlpath1, "flow", "max", IDS, name="checkersphere,\nflow blending", pos="", scene="sphere"),

    ResultSet(tlpath2, "regular", "max", IDS, name="square\nroom", scene="square"),
#    ResultSet(tlpath2, "flow", "max", IDS, name="square\nroom", scene="square"),

#    ResultSet(tlpath3, "regular", "max", IDS, name="oblong\nroom", scene="oblong"),
#    ResultSet(tlpath3, "flow", "max", IDS, name="oblong\nroom", scene="oblong"),
        ]

#comparing viewpoint densities
tlpath1 = "../../data/captures/thesis_selection/square_synth_room/2x2v2_offset_new/"
tlpath2 = "../../data/captures/thesis_selection/square_synth_room/6x6_offset_new/"
tlpath3 = "../../data/captures/thesis_selection/square_synth_room/12x12_offset_new/"
pos_name = "dens"
pos_res_sets = [
#    ResultSet(tlpath1, "regular", "max", IDS, name="2x2", scene="2x2"),
#    ResultSet(tlpath2, "regular", "max", IDS, name="6x6", scene="6x6"),
    ResultSet(tlpath3, "regular", "max", IDS, name="12x12", scene="12x12"),
#    ResultSet(tlpath1, "flow", "max", IDS, name="2x2", scene="2x2"),
#    ResultSet(tlpath2, "flow", "max", IDS, name="6x6", scene="6x6"),
    ResultSet(tlpath3, "flow", "max", IDS, name="12x12", scene="12x12"),
        ]
'''

'''
#dense scene
tlpath = "../../data/captures/thesis_selection/square_synth_room/6x6_dense/"
name = "dense"
IDS = list(map(str, list(range(25*25))))
pos_res_sets = [
#    ResultSet(tlpath, "baseline", "max", IDS, name="square\nroom", scene="square"),
    ResultSet(tlpath, "regular", "max", IDS, name="square\nroom", scene="square"),
    ResultSet(tlpath, "flow", "max", IDS, name="square\nroom", scene="square"),
    ]
'''



'''
#real scene baseline vs regular vs flow
tlpath = "../../data/captures/thesis_selection/sw/5x5_random/"
pos_name = "Real scene max vps"
pos_res_sets = [
#            ResultSet(tlpath, "baseline", "max", IDS[:13], scene="sw"),
            ResultSet(tlpath, "regular", "max", IDS[:13], scene="sw"),
            ResultSet(tlpath, "flow", "max", IDS[:13], scene="sw"),
        ]
'''
