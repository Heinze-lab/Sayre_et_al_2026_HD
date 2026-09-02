import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm
from cx import extrapolation


CELLS = pl.read_csv("../data/connectome/fly-cells.csv").with_columns(columns = pl.col("subtype").str.extract_all("[LR]\\d+")).filter(pl.col("subtype").is_in({"L4R6", "L6R4", "L7R3"}).not_()).with_columns(pl.row_index())

# Define neuropils
ROIS = ["PB", "EB"]

COLUMNS_BY_ROI = {
    "EB": ["c1R", "c1L", "c2R", "c2L", "c3R", "c3L", "c4R", "c4L", "c5R", "c5L", "c6R", "c6L", "c7R", "c7L", "c8R", "c8L", "c9R", "c9L", "c10R", "c10L", "c11R", "c11L", "c12R", "c12L", "c13R", "c13L", "c14R", "c14L", "c15R", "c15L", "c16R", "c16L"],
    "PB": [
        "R9R", "R9L", "R8R", "R8L", "R7R", "R7L", "R6R", "R6L", "R5R", "R5L", "R4R", "R4L", "R3R", "R3L", "R2R", "R2L", "R1R", "R1L",
        "L1R", "L1L", "L2R", "L2L", "L3R", "L3L", "L4R", "L4L", "L5R", "L5L", "L6R", "L6L", "L7R", "L7L", "L8R", "L8L", "L9R", "L9L",
    ],
}

COLUMNS = COLUMNS_BY_ROI["EB"] + COLUMNS_BY_ROI["PB"]

MIRROR_SIDE = {"L": "R", "R": "L"}

def mirror_column(col: str):
    if col[0] in MIRROR_SIDE.keys():
        return f"{MIRROR_SIDE[col[0]]}{col[1:-1]}{MIRROR_SIDE[col[-1]]}"
    else:
        n = 17 - int(col[1:-1])
        return f"c{n}{MIRROR_SIDE[col[-1]]}"


# Define projections
PROJECTIONS = {
    "PB": {
        "EPG": {
        }
    },
    "EB": {
        "EPG": {
            #"R9": ["c15"],
            "R8": ["c2R", "c2L"],
            "R7": ["c4R", "c4L"],
            "R6": ["c6R", "c6L"],
            "R5": ["c8R", "c8L"],
            "R4": ["c10R", "c10L"],
            "R3": ["c12R", "c12L"],
            "R2": ["c14R", "c14L"],
            "R1": ["c16R", "c16L"],
            "L1": ["c1R", "c1L"],
            "L2": ["c3R", "c3L"],
            "L3": ["c5R", "c5L"],
            "L4": ["c7R", "c7L"],
            "L5": ["c9R", "c9L"],
            "L6": ["c11R", "c11L"],
            "L7": ["c13R", "c13L"],
            "L8": ["c15R", "c15L"],
            #"L9": ["c16"],
        },
        "PEG": {
            #"R9": ["c15"],
            "R8": ["c2R", "c2L"],
            "R7": ["c4R", "c4L"],
            "R6": ["c6R", "c6L"],
            "R5": ["c8R", "c8L"],
            "R4": ["c10R", "c10L"],
            "R3": ["c12R", "c12L"],
            "R2": ["c14R", "c14L"],
            "R1": ["c16R", "c16L"],
            "L1": ["c1R", "c1L"],
            "L2": ["c3R", "c3L"],
            "L3": ["c5R", "c5L"],
            "L4": ["c7R", "c7L"],
            "L5": ["c9R", "c9L"],
            "L6": ["c11R", "c11L"],
            "L7": ["c13R", "c13L"],
            "L8": ["c15R", "c15L"],
            #"L9": ["c16"],
        },
        #"PEN_a": {
        #    "R9": ["c2L"],
        #    "R8": ["c4L"],
        #    "R7": ["c6L"],
        #    "R6": ["c8L"],
        #    "R5": ["c10L"],
        #    "R4": ["c12L"],
        #    "R3": ["c14L"],
        #    "R2": ["c16L"],
        #    "L2": ["c1R"],
        #    "L3": ["c3R"],
        #    "L4": ["c5R"],
        #    "L5": ["c7R"],
        #    "L6": ["c9R"],
        #    "L7": ["c11R"],
        #    "L8": ["c13R"],
        #    "L9": ["c15R"],
        #},
        #"PEN_b": {
        #    "R9": ["c2L"],
        #    "R8": ["c4L"],
        #    "R7": ["c6L"],
        #    "R6": ["c8L"],
        #    "R5": ["c10L"],
        #    "R4": ["c12L"],
        #    "R3": ["c14L"],
        #    "R2": ["c16L"],
        #    "L2": ["c1R"],
        #    "L3": ["c3R"],
        #    "L4": ["c5R"],
        #    "L5": ["c7R"],
        #    "L6": ["c9R"],
        #    "L7": ["c11R"],
        #    "L8": ["c13R"],
        #    "L9": ["c15R"],
        #},
        # Center of column:
        "PEN_a": {
            "R9": ["c3R", "c2L"],
            "R8": ["c5R", "c4L"],
            "R7": ["c7R", "c6L"],
            "R6": ["c9R", "c8L"],
            "R5": ["c11R", "c10L"],
            "R4": ["c13R", "c12L"],
            "R3": ["c15R", "c14L"],
            "R2": ["c1R", "c16L"],
            "L2": ["c1R", "c16L"],
            "L3": ["c3R", "c2L"],
            "L4": ["c5R", "c4L"],
            "L5": ["c7R", "c6L"],
            "L6": ["c9R", "c8L"],
            "L7": ["c11R", "c10L"],
            "L8": ["c13R", "c12L"],
            "L9": ["c15R", "c14L"],
        },
        "PEN_b": {
            "R9": ["c3R", "c2L"],
            "R8": ["c5R", "c4L"],
            "R7": ["c7R", "c6L"],
            "R6": ["c9R", "c8L"],
            "R5": ["c11R", "c10L"],
            "R4": ["c13R", "c12L"],
            "R3": ["c15R", "c14L"],
            "R2": ["c1R", "c16L"],
            "L2": ["c1R", "c16L"],
            "L3": ["c3R", "c2L"],
            "L4": ["c5R", "c4L"],
            "L5": ["c7R", "c6L"],
            "L6": ["c9R", "c8L"],
            "L7": ["c11R", "c10L"],
            "L8": ["c13R", "c12L"],
            "L9": ["c15R", "c14L"],
        },
        # On top of EPGs:
        #"PEN_a": {
        #    "R9": ["c2R", "c2L"],
        #    "R8": ["c4R", "c4L"],
        #    "R7": ["c6R", "c6L"],
        #    "R6": ["c8R", "c8L"],
        #    "R5": ["c10R", "c10L"],
        #    "R4": ["c12R", "c12L"],
        #    "R3": ["c14R", "c14L"],
        #    "R2": ["c16R", "c16L"],
        #    "L2": ["c1R", "c1L"],
        #    "L3": ["c3R", "c3L"],
        #    "L4": ["c5R", "c5L"],
        #    "L5": ["c7R", "c7L"],
        #    "L6": ["c9R", "c9L"],
        #    "L7": ["c11R", "c11L"],
        #    "L8": ["c13R", "c13L"],
        #    "L9": ["c15R", "c15L"],
        #},
        #"PEN_b": {
        #    "R9": ["c2R", "c2L"],
        #    "R8": ["c4R", "c4L"],
        #    "R7": ["c6R", "c6L"],
        #    "R6": ["c8R", "c8L"],
        #    "R5": ["c10R", "c10L"],
        #    "R4": ["c12R", "c12L"],
        #    "R3": ["c14R", "c14L"],
        #    "R2": ["c16R", "c16L"],
        #    "L2": ["c1R", "c1L"],
        #    "L3": ["c3R", "c3L"],
        #    "L4": ["c5R", "c5L"],
        #    "L5": ["c7R", "c7L"],
        #    "L6": ["c9R", "c9L"],
        #    "L7": ["c11R", "c11L"],
        #    "L8": ["c13R", "c13L"],
        #    "L9": ["c15R", "c15L"],
        #},
    },
    "NO": {
        "PEN_a": {
            "R9": ["right"],
            "R8": ["right"],
            "R7": ["right"],
            "R6": ["right"],
            "R5": ["right"],
            "R4": ["right"],
            "R3": ["right"],
            "R2": ["right"],
            "L2": ["left"],
            "L3": ["left"],
            "L4": ["left"],
            "L5": ["left"],
            "L6": ["left"],
            "L7": ["left"],
            "L8": ["left"],
            "L9": ["left"],
        },
        "PEN_b": {
            "R9": ["right"],
            "R8": ["right"],
            "R7": ["right"],
            "R6": ["right"],
            "R5": ["right"],
            "R4": ["right"],
            "R3": ["right"],
            "R2": ["right"],
            "L2": ["left"],
            "L3": ["left"],
            "L4": ["left"],
            "L5": ["left"],
            "L6": ["left"],
            "L7": ["left"],
            "L8": ["left"],
            "L9": ["left"],
        },
    },
}

auto_populate = { roi: set() for roi in PROJECTIONS.keys() }
for cell_type, subtype, _ in CELLS.group_by("type", "subtype").agg(pl.len()).rows():
    for roi in PROJECTIONS.keys():
        if cell_type not in PROJECTIONS[roi] or subtype not in PROJECTIONS[roi][cell_type]:
            auto_populate[roi].add((cell_type, subtype))

# automatically populate rest of bridge
for index, _, cell_type, subtype, output_columns in CELLS.rows():
    if (cell_type, subtype) in auto_populate["PB"]:
        if cell_type not in PROJECTIONS["PB"]:
            PROJECTIONS["PB"][cell_type] = {}
        out = set([c + "R" for c in output_columns]) | set([c + "L" for c in output_columns])
        #if cell_type == "EPG":
        #    out -= {"R9", "L9"}
        PROJECTIONS["PB"][cell_type][subtype] = list(out) #list(set(output_columns) - {"R9", "L9"})

# load ground truth
conns = pl.read_csv("../data/connectome/fly_hemibrain_all_hd_cells_conntable.csv")

W_ground = np.zeros((len(ROIS), len(CELLS), len(CELLS)))
types = CELLS["type"].unique(maintain_order=True)
for r, roi in enumerate(ROIS):
    indexed_counts = (
        conns
        .filter((pl.col("roi") == roi)) # & (pl.col("type_pre") == "EPG") & (pl.col("type_post") == "EPG"))
        .select("bodyId_pre", "bodyId_post", "count")
        .join(CELLS.select("bodyId", subtype_pre = "subtype", index_pre = "index"), left_on="bodyId_pre", right_on="bodyId")
        .join(CELLS.select("bodyId", subtype_post = "subtype", index_post = "index"), left_on="bodyId_post", right_on="bodyId")
    )
    for i, j, count in indexed_counts.select("index_pre", "index_post", "count").rows():
        W_ground[r,i,j] += count

# Construct reference weight matrix within sample
sample_conns = pl.read_csv("../data/connectome/fly_bee_roi_conntable.csv").filter(pl.col("roi_pre") == pl.col("roi_post")).with_columns(roi = pl.col("roi_pre"))

W_sample = np.zeros((len(ROIS), len(CELLS), len(CELLS)))
types = CELLS["type"].unique(maintain_order=True)
for r, roi in enumerate(ROIS):
    indexed_counts = (
        sample_conns
        .filter((pl.col("roi") == roi)) # & (pl.col("type_pre") == "EPG") & (pl.col("type_post") == "EPG"))
        .select("bodyId_pre", "bodyId_post", "count")
        .join(CELLS.select("bodyId", subtype_pre = "subtype", index_pre = "index"), left_on="bodyId_pre", right_on="bodyId")
        .join(CELLS.select("bodyId", subtype_post = "subtype", index_post = "index"), left_on="bodyId_post", right_on="bodyId")
    )
    for i, j, count in indexed_counts.select("index_pre", "index_post", "count").rows():
        W_sample[r,i,j] += count

#SAMPLE = {"L3R", "L3L", "L4R", "L4L", "L5R", "L5L", "L6R", "c2R", "c2L", "c3R"}
SAMPLE = {"L3R", "L3L", "L4R", "L4L", "L5R", "L5L", "L6R", "c1L", "c2R", "c2L", "c3R"} #, "c3L"}
MIRRORED_SAMPLE = { mirror_column(c) for c in SAMPLE }
EXTENDED_SAMPLE = SAMPLE | MIRRORED_SAMPLE
