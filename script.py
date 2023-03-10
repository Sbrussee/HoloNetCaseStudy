import HoloNet as hn

import os
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import torch

#Load example Visium dataset (24,923 genes, 3798 spots)
visium_example_dataset = hn.pp.load_brca_visium_10x()
#Plot cell type percentages
hn.pl.plot_cell_type_proportion(visium_example_dataset, plot_cell_type='stroma')

#Cell type labels per spot
sc.pl.spatial(visium_example_dataset, color=['cell_type'], size=1.4, alpha=0.7,
palette=hn.brca_default_color_celltype)

#Load human L-R database from CellChatDB, Connectome also possible
interaction_db, cofactor_db, complex_db = hn.pp.load_lr_df(human_or_mouse='human')
#Filter LR-pairs by occuring at a percentage of cells (0.3)
expressed_lr_df = hn.pp.get_expressed_lr_df(interaction_db, complex_db, visium_example_dataset,
                                            expressed_prop = 0.15)
print("LR dataframe shape: "+str(expressed_lr_df.shape))

"""
As ligand molecules from a single source can only cover a limited region,
We select a range around the ligand denoted as w_best
"""
#We can select the default one for visium, we could specify it ourselves too
w_best = hn.tl.default_w_visium()
hn.pl.select_w(visium_example_dataset, w_best=w_best)

#Now we can build the multi-view CCC network:
#We construct a expression dataframe
elements_expr_df_dict = hn.tl.elements_expr_df_calculate(expressed_lr_df, complex_db,
                                                        cofactor_db, visium_example_dataset)
#Now we compute the tensor of communication events
ce_tensor = hn.tl.compute_ce_tensor(expressed_lr_df, w_best, elements_expr_df_dict, visium_example_dataset)
#We can then filter the edges with low specifities
filtered_ce_tensor = hn.tl.filter_ce_tensor(ce_tensor, visium_example_dataset, expressed_lr_df,
                                            elements_expr_df_dict, w_best)

#Now that we have our views, we can visualize each CE both on cell-level as well as on cell-type-level
#We can use either degree or eigenvector centrality as CE strength per cell/spot
#For example, let's see it for TGFB1:(TGFBR1+TGFBR2)
