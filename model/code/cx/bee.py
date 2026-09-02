import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm
from cx import extrapolation

CELLS = pl.read_parquet("../data/connectome/bee-cells.parquet")

# Define neuropils
ROIS = ["PB", "EB"]

COLUMNS_BY_ROI = {
    "EB": ["c1R", "c1L", "c2R", "c2L", "c3R", "c3L", "c4R", "c4L", "c5R", "c5L", "c6R", "c6L", "c7R", "c7L", "c8R", "c8L", "c9R", "c9L"],
    "PB": [
        "R10R", "R10L", "R9R", "R9L", "R8R", "R8L", "R7R", "R7L", "R6R", "R6L", "R5R", "R5L", "R4R", "R4L", "R3R", "R3L", "R2R", "R2L", "R1R", "R1L",
        "L1R", "L1L", "L2R", "L2L", "L3R", "L3L", "L4R", "L4L", "L5R", "L5L", "L6R", "L6L", "L7R", "L7L", "L8R", "L8L", "L9R", "L9L", "L10R", "L10L",
    ],
}

COLUMNS = COLUMNS_BY_ROI["EB"] + COLUMNS_BY_ROI["PB"]
 
MIRROR_SIDE = {"L": "R", "R": "L"}

def mirror_column(col: str):
    if col[0] in MIRROR_SIDE.keys():
        return f"{MIRROR_SIDE[col[0]]}{col[1:-1]}{MIRROR_SIDE[col[-1]]}"
    else:
        n = 10 - int(col[1:-1])
        return f"c{n}{MIRROR_SIDE[col[-1]]}"


# Define projections
PROJECTIONS = {
    "PB": {
        "EPG": {
            "R1": ["R1L", "R1R", "L1L", "L1R"],
            "L1": ["R1L", "R1R", "L1L", "L1R"],
        },
        "PEG": {
            "R1": ["R1L", "R1R", "L1L", "L1R"],
            "L1": ["R1L", "R1R", "L1L", "L1R"],
        },
        "PEN_a": {
            "R9": ["R9R", "R9L", "R10R", "R10L"],
            "L9": ["L9R", "L9L", "L10R", "L10L"],
        },
        "PEN_b": {
            "R9": ["R9R", "R9L", "R10R", "R10L"],
            "L9": ["L9R", "L9L", "L10R", "L10L"],
        }
    },
    "EB": {
        "EPG": {
            "R9": ["c1R", "c1L"],
            "R8": ["c2R", "c2L"],
            "R7": ["c3R", "c3L"],
            "R6": ["c4R", "c4L"],
            "R5": ["c5R", "c5L"],
            "R4": ["c6R", "c6L"],
            "R3": ["c7R", "c7L"],
            "R2": ["c8R", "c8L"],
            "R1": ["c9R", "c9L"],
            "L1": ["c1R", "c1L"],
            "L2": ["c2R", "c2L"],
            "L3": ["c3R", "c3L"],
            "L4": ["c4R", "c4L"],
            "L5": ["c5R", "c5L"],
            "L6": ["c6R", "c6L"],
            "L7": ["c7R", "c7L"],
            "L8": ["c8R", "c8L"],
            "L9": ["c9R", "c9L"],
        },
        "PEG": {
            #"R9": ["c1R", "c1L"],
            "R8": ["c2R", "c2L"],
            "R7": ["c3R", "c3L"],
            "R6": ["c4R", "c4L"],
            "R5": ["c5R", "c5L"],
            "R4": ["c6R", "c6L"],
            "R3": ["c7R", "c7L"],
            "R2": ["c8R", "c8L"],
            "R1": ["c9R", "c9L"],
            "L1": ["c1R", "c1L"],
            "L2": ["c2R", "c2L"],
            "L3": ["c3R", "c3L"],
            "L4": ["c4R", "c4L"],
            "L5": ["c5R", "c5L"],
            "L6": ["c6R", "c6L"],
            "L7": ["c7R", "c7L"],
            "L8": ["c8R", "c8L"],
            #"L9": ["c9R", "c9L"],
        },
        "PEN_a": {
            "R9": ["c1R", "c1L", "c2R", "c2L"],
            "R8": ["c3R", "c3L"],
            "R7": ["c4R", "c4L"],
            "R6": ["c5R", "c5L"],
            "R5": ["c6R", "c6L"],
            "R4": ["c7R", "c7L"],
            "R3": ["c8R", "c8L"],
            "R2": ["c9R", "c9L", "c8L", "c8R"],
            "L2": ["c1R", "c1L", "c2L", "c2R"],
            "L3": ["c2R", "c2L"],
            "L4": ["c3R", "c3L"],
            "L5": ["c4R", "c4L"],
            "L6": ["c5R", "c5L"],
            "L7": ["c6R", "c6L"],
            "L8": ["c7R", "c7L"],
            "L9": ["c8R", "c8L", "c9R", "c9L"],
        },
        "PEN_b": {
            "R9": ["c1R", "c1L", "c2R", "c2L"],
            "R8": ["c3R", "c3L"],
            "R7": ["c4R", "c4L"],
            "R6": ["c5R", "c5L"],
            "R5": ["c6R", "c6L"],
            "R4": ["c7R", "c7L"],
            "R3": ["c8R", "c8L"],
            "R2": ["c9R", "c9L"],
            "L2": ["c1R", "c1L"],
            "L3": ["c2R", "c2L"],
            "L4": ["c3R", "c3L"],
            "L5": ["c4R", "c4L"],
            "L6": ["c5R", "c5L"],
            "L7": ["c6R", "c6L"],
            "L8": ["c7R", "c7L"],
            "L9": ["c8R", "c8L", "c9R", "c9L"],
        },
    },
}

auto_populate = { roi: set() for roi in PROJECTIONS.keys() }
for cell_type, subtype, _ in CELLS.group_by("type", "subtype").agg(pl.len()).rows():
    for roi in PROJECTIONS.keys():
        if cell_type not in PROJECTIONS[roi] or subtype not in PROJECTIONS[roi][cell_type]:
            auto_populate[roi].add((cell_type, subtype))

# automatically populate rest of bridge
for index, cell_type, subtype, output_columns in CELLS.rows():
    if (cell_type, subtype) in auto_populate["PB"]:
        if cell_type not in PROJECTIONS["PB"]:
            PROJECTIONS["PB"][cell_type] = {}
        #out = output_columns
        out = set([c + "R" for c in output_columns]) | set([c + "L" for c in output_columns])
        #if cell_type == "EPG":
        PROJECTIONS["PB"][cell_type][subtype] = list(out) #list(set(output_columns) - {"R9", "L9"})

PROJECTIONS

cell_types = CELLS["type"].unique()
conns = (
    pl.read_csv("../data/connectome/bee_conntable.csv")
    .with_columns(
        type_pre_col = pl.col("type_pre_col").replace({ "EPG_L_R1L1": "EPG_L1", "EPG_R_R1L1": "EPG_R1" }),
        type_post_col = pl.col("type_post_col").replace({ "EPG_L_R1L1": "EPG_L1", "EPG_R_R1L1": "EPG_R1" }),
    )
    .filter(pl.col("type_pre").is_in(cell_types.to_list()) & pl.col("type_post").is_in(cell_types.to_list()))
)

left = CELLS.with_columns(subindex = pl.col("index").cum_count().over(("type", "subtype")) - 1)

conn_cells = (
    conns
    .select(id = "pre_skid", type = "type_pre", name = "type_pre_col").unique(maintain_order=True).join(
        conns.select(id = "post_skid", type = "type_post", name = "type_post_col").unique(maintain_order=True),
        on="id",
        how="full",
        maintain_order="left_right",
    )
    .select(
        id = pl.coalesce("id", "id_right"),
        type = pl.coalesce("type", "type_right"),
        name = pl.coalesce("name", "name_right"),
    )
    .with_columns(subtype = pl.col("name").str.extract(r"([LR]\d+)+$", 0))
    .drop("name")
    .with_columns(pl.row_index())
    .with_columns(subindex = pl.col("index").cum_count().over("type", "subtype") - 1)
    .join(left, on=("type", "subtype", "subindex"), maintain_order="left_right")
    .select("id", "type", "subtype", "columns", index = "index_right")
)

indexed_counts = (
    conns
    .select("pre_skid", "pre_name", "post_name", "post_skid", "count", "roi")
    .join(conn_cells.select("id", type_pre = "type", subtype_pre="subtype", cols_pre = "columns", index_pre = "index"), left_on="pre_skid", right_on="id", maintain_order="left_right")
    .join(conn_cells.select("id", type_post = "type", subtype_post="subtype", cols_post = "columns", index_post = "index"), left_on="post_skid", right_on="id", maintain_order="left_right")
)

W_sample = np.zeros((len(ROIS), len(CELLS), len(CELLS)))

roi_index = {"PB": 0, "EB": 1}
for idx_pre, idx_post, pre_name, post_name, type_pre, type_post, subtype_pre, subtype_post, count, roi in indexed_counts.select("index_pre", "index_post", "pre_name", "post_name", "type_pre", "type_post", "subtype_pre", "subtype_post", "count", "roi").rows():
    #print(pre_name, type_pre, subtype_pre, " - ", post_name, type_post, subtype_post)
    W_sample[roi_index[roi], idx_pre, idx_post] += count

SAMPLE = {"L2L", "L3R", "L3L", "L4R", "L4L", "L5R", "L5L", "L6R", "c1R", "c1L", "c2R", "c2L"}
MIRRORED_SAMPLE = { mirror_column(c) for c in SAMPLE }
EXTENDED_SAMPLE = SAMPLE | MIRRORED_SAMPLE

