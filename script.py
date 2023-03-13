import HoloNet as hn

import os
from os import path
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import torch

dirpath = os.getcwd()
outpath = dirpath + "/output"
if not os.path.exists(outpath):
    os.mkdir("output")

#Load example Visium dataset (24,923 genes, 3798 spots)
visium_example_dataset = hn.pp.load_brca_visium_10x()
name = 'brca_visium'

#Plot cell type percentages
hn.pl.plot_cell_type_proportion(visium_example_dataset, plot_cell_type='stroma')

#Cell type labels per spot
sc.pl.spatial(visium_example_dataset, color=['cell_type'], size=1.4, alpha=0.7,
palette=hn.brca_default_color_celltype, save="spatial.png")

#Load human L-R database from CellChatDB, Connectome also possible
interaction_db, cofactor_db, complex_db = hn.pp.load_lr_df(human_or_mouse='human')
#Filter LR-pairs by occuring at a percentage of cells (0.3)
if path.exists("output/expressed_lr_df_"+name+".csv"):
    expressed_lr_df = pd.read_csv("output/expressed_lr_df_" + name + ".csv")
else:
    expressed_lr_df = hn.pp.get_expressed_lr_df(interaction_db, complex_db, visium_example_dataset,
                                                expressed_prop = 0.15)
    expressed_lr_df.to_csv("output/expressed_lr_df_" + name + ".csv")
print("LR dataframe shape: "+str(expressed_lr_df.shape))

"""
As ligand molecules from a single source can only cover a limited region,
We select a range around the ligand denoted as w_best
"""
#We can select the default one for visium, we could specify it ourselves too
w_best = hn.tl.default_w_visium(visium_example_dataset)
hn.pl.select_w(visium_example_dataset, w_best=w_best)

#Now we can build the multi-view CCC network:
#We construct a expression dataframe
if path.exists("output/elements_expr_df_"+name+".csv"):
    elements_expr_df_dict = pd.read_csv("output/elements_expr_df_" + name + ".csv").to_dict()
else:
    elements_expr_df_dict = hn.tl.elements_expr_df_calculate(expressed_lr_df, complex_db,
                                                            cofactor_db, visium_example_dataset)
    pd.DataFrame.from_dict(elements_expr_df_dict).to_csv("output/elements_expr_df_" + name + ".csv")
#elements_expr_df_dict.to_csv("output/elements_expr_df_" + name + ".csv")
print("Expr matrix shape: "+str(elements_expr_df_dict.shape))

#Now we compute the tensor of communication events
ce_tensor = hn.tl.compute_ce_tensor(expressed_lr_df, w_best, elements_expr_df_dict, visium_example_dataset)
#We can then filter the edges with low specifities
if path.exists("outputs/filtered_ce_tensor_"+name+".csv"):
    filtered_ce_tensor = pd.read_csv("output/filtered_ce_tensor_" + name + ".csv")
else:
    filtered_ce_tensor = hn.tl.filter_ce_tensor(ce_tensor, visium_example_dataset, expressed_lr_df,
                                                elements_expr_df_dict, w_best)
    filtered_ce_tensor.to_csv("output/filtered_ce_tensor_" + name + ".csv")

#Now that we have our views, we can visualize each CE both on cell-level as well as on cell-type-level
#We can use either degree or eigenvector centrality as CE strength per cell/spot
#For example, let's see it for TGFB1:(TGFBR1+TGFBR2)

hn.pl.ce_hotspot_plot(filtered_ce_tensor, visium_example_dataset,
lr_df=expressed_lr_df, plot_lr='TGFB1:(TGFBR1+TGFBR2)', fname='output/ce_hotspot')

#Now based on eigenvector centrality

hn.pl.ce_hotspot_plot(filtered_ce_tensor, visium_example_dataset,
lr_df=expressed_lr_df, plot_lr='TGFB1:(TGFBR1+TGFBR2)', fname='output/ce_hotspot_eigenvector',
centrality_measure='eigenvector')

#We can also plot the cel-type CE network.
#for this, we need to load the cell-type percentages per spot
cell_type_mat, cell_type_names = hn.pr.get_continous_cell_type_tensor(visium_example_dataset,
                                                                      continous_cell_type_slot='predicted_cell_type')

_ = hn.pl.ce_cell_type_network_plot(filtered_ce_tensor, cell_type_mat, cell_type_names,
lr_df=expressed_lr_df, plot_lr="TGFB1:(TGFBR1+TGFBR2)", edge_thres=0.2,
palette=hn.brca_default_color_celltype, fname='output/cell_type_network')

#We can perform agglomerative clustering for the igand-receptor pairs based on the centrality measures.
cell_cci_centrality = hn.tl.compute_ce_network_eigenvector_centrality(filter_ce_tensor)
clustered_expressed_LR_df, _ = hn.tl.cluster_lr_based_on_ce(filtered_ce_tensor, visium_example_dataset, expressed_lr_df,
w_best=w_best, cell_cci_centrality=cell_cci_centrality)

#Now plot a dendogram using this clustering
hn.pl.lr_clustering_dendogram(_, expressed_lr_df, ['TGFB1:(TGFBR1+TGFBR2)'],
dflt_col="#333333",fname="output/clust_dendogram")

#We can also plot the general CE hotspots for each ligand-receptor cluster
hn.pl.lr_cluster_ce_hotspot_plot(lr_df=clustered_expressed_LR_df,
cell_cci_centrality=cell_cci_centrality,
adata=visium_example_dataset, fname='output/general_ce_hotspot')
