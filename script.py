import HoloNet as hn

import sys
import os
from os import path
import requests
import argparse
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import torch

#Remove warnings from output
import warnings
warnings.filterwarnings('ignore')
hn.set_figure_params(tex_fonts=False)
sc.settings.figdir = './figures/'

#Get current directory, make sure output directory exists
dirpath = os.getcwd()
outpath = dirpath + "/output"
if not os.path.exists(outpath):
    os.mkdir("output")

#Check if GPU is available
if torch.cuda.is_available():
    print("GPU available.")
else:
    print("GPU unavailable")

#Build CL argument parser
parser = argparse.ArgumentParser(prog="GNN-Framework",
                                description="Framework for testing GNN-based CCC inference methods")
parser.add_argument("-d", '--dataset', default="brca_visium", help='Which dataset to analyze')
parser.add_argument('-hn', '--holonet', action='store_true', help='Whether to apply HoloNet')
parser.add_argument('-g', '--genes', help='List of target genes to query')
parser.add_argument('-p', '--pairs', help='List of ligand receptor pairs to query')
args = parser.parse_args()

class holonet_pipeline:
    """
        - dataset: AnnData dataset to analyze
        - organism: Organism of dataset: mouse/human
        - list_of_target_genes: List of target genes to train models for
        - list_of_target_lr : List of LR-pairs to visualize
        - name: Name of the dataset (for use in plotting)
    """
    def __init__(self, dataset, organism, list_of_target_genes=[], list_of_target_lr=[], name=""):
        self.dataset = dataset
        self.organism = organism
        self.list_of_target_genes = list_of_target_genes
        self.list_of_target_lr = list_of_target_lr
        self.name = name

        self.visualize_dataset()
        #Load the Ligand-Receptor matrix
        self.load_lr_df()
        #Create the Cellular Event tensor
        self.load_ce_tensor()
        #Visualize each LR-pair
        if len(list_of_target_genes) < 1:
            for pair in self.expressed_lr_df['LR_Pair'].to_list():
                self.visualize_ce_tensors(pair)
        else:
            for pair in list_of_target_lr:
                self.visualize_ce_tensors(pair)

        self.preprocessing_for_gcn_model()
        model_per_gene = {}
        if len(list_of_target_genes) < 1:
            self.multitarget_training(self.used_gene_list)
            #Train a model per gene
            for gene in self.used_gene_list:
                model_per_gene[gene] = self.train_gcn_model(gene)
        else:
            self.multitarget_training(list_of_target_genes)
            #Train a model per gene
            for gene in list_of_target_genes:
                model_per_gene[gene] = self.train_gcn_model(gene)

        for gene, model in model_per_gene.items():
            self.visualize_gcn_output(gene, model)



    def visualize_dataset(self):
        print("Visualizing dataset...")
        #Plot cell type percentages
        hn.pl.plot_cell_type_proportion(dataset, plot_cell_type='stroma', fname="cell_type_proportions.png")

        #Cell type labels per spot
        sc.pl.spatial(dataset, color=['cell_type'], size=1.4, alpha=0.7,
        palette=hn.brca_default_color_celltype, save="spatial.png")


    def load_lr_df(self):
        print("Load expression matrix...")
        #Load human L-R database from CellChatDB, Connectome also possible
        self.interaction_db, self.cofactor_db, self.complex_db = hn.pp.load_lr_df(human_or_mouse=self.organism)
        #Filter LR-pairs by occuring at a percentage of cells (0.3)
        if path.exists("output/expressed_lr_df_"+self.name+".csv"):
            self.expressed_lr_df = pd.read_csv("output/expressed_lr_df_" + self.name + ".csv")
        else:
            self.expressed_lr_df = hn.pp.get_expressed_lr_df(self.interaction_db, self.complex_db,
                                                             self.dataset, expressed_prop=0.15)
            self.expressed_lr_df.to_csv("output/expressed_lr_df_" + self.name + ".csv")
        print("LR dataframe shape: "+str(self.expressed_lr_df.shape))

    def load_ce_tensor(self):
        print("creating CE tensor...")
        """
        As ligand molecules from a single source can only cover a limited region,
        We select a range around the ligand denoted as w_best
        """
        #We can select the default one for visium, we could specify it ourselves too
        self.w_best = hn.tl.default_w_visium(self.dataset)
        hn.pl.select_w(self.dataset, w_best=self.w_best)

        #Now we can build the multi-view CCC network:
        #We construct a expression dataframe
        if path.exists("output/elements_expr_df_dict_"+self.name+".pkl"):
            with open("output/elements_expr_df_dict_"+self.name+".pkl", 'rb') as f:
                self.elements_expr_df_dict = pickle.load(f)
        else:
            self.elements_expr_df_dict = hn.tl.elements_expr_df_calculate(self.expressed_lr_df, self.complex_db,
                                                                    self.cofactor_db, self.dataset)
            with open("output/elements_expr_df_dict_"+self.name+".pkl", 'wb') as f:
                pickle.dump(self.elements_expr_df_dict, f)

        #Now we compute the tensor of communication events
        if path.exists("output/ce_tensor_"+self.name+".pkl"):
            with open("output/ce_tensor_"+self.name+".pkl", 'rb') as f:
                ce_tensor = pickle.load(f)
        else:
            ce_tensor = hn.tl.compute_ce_tensor(self.expressed_lr_df, self.w_best,
                                                 self.elements_expr_df_dict, self.dataset)
            with open('output/ce_tensor_'+self.name+".pkl", 'wb') as f:
                pickle.dump(ce_tensor, f)
        #We can then filter the edges with low specifities
        if path.exists("output/filtered_ce_tensor_"+self.name+".pkl"):
            with open("output/filtered_ce_tensor_"+self.name+".pkl", 'rb') as f:
                self.filtered_ce_tensor = pickle.load(f)
        else:
            self.filtered_ce_tensor = hn.tl.filter_ce_tensor(ce_tensor, self.dataset, self.expressed_lr_df,
                                                             self.elements_expr_df_dict, self.w_best)
            with open("output/filtered_ce_tensor_"+self.name+".pkl", 'wb') as f:
                pickle.dump(self.filtered_ce_tensor, f)

    def visualize_ce_tensors(self, target_lr="TGFB1:(TGFBR1+TGFBR2)"):
        print("Visualizing CE tensors...")
        #Now that we have our views, we can visualize each CE both on cell-level as well as on cell-type-level
        #We can use either degree or eigenvector centrality as CE strength per cell/spot
        #For example, let's see it for TGFB1:(TGFBR1+TGFBR2)

        hn.pl.ce_hotspot_plot(self.filtered_ce_tensor, self.dataset,
        lr_df=self.expressed_lr_df, plot_lr=target_lr, fname='output/ce_hotspot_'+self.name+"_"+target_lr+".png")

        #Now based on eigenvector centrality
        hn.pl.ce_hotspot_plot(self.filtered_ce_tensor, self.dataset,
        lr_df=self.expressed_lr_df, plot_lr=target_lr, fname='output/ce_hotspot_eigenvector_'+self.name+"_"+target_lr+".png",
        centrality_measure='eigenvector')

        #We can also plot the cel-type CE network.
        #for this, we need to load the cell-type percentages per spot
        self.cell_type_mat, self.cell_type_names = hn.pr.get_continuous_cell_type_tensor(self.dataset,
                                                                              continuous_cell_type_slot='predicted_cell_type')

        _ = hn.pl.ce_cell_type_network_plot(self.filtered_ce_tensor, self.cell_type_mat, self.cell_type_names,
        lr_df=self.expressed_lr_df, plot_lr=target_lr, edge_thres=0.2,
        palette=hn.brca_default_color_celltype, fname='output/cell_type_network_'+self.name+"_"+target_lr+".png")

        #We can perform agglomerative clustering for the igand-receptor pairs based on the centrality measures.
        cell_cci_centrality = hn.tl.compute_ce_network_eigenvector_centrality(self.filtered_ce_tensor)
        self.clustered_expressed_LR_df, _ = hn.tl.cluster_lr_based_on_ce(self.filtered_ce_tensor, self.dataset, self.expressed_lr_df,
        w_best=self.w_best, cell_cci_centrality=cell_cci_centrality)

        #Now plot a dendogram using this clustering
        hn.pl.lr_clustering_dendrogram(_, self.expressed_lr_df, [target_lr],
        dflt_col="#333333",fname="output/clust_dendogram_"+self.name+"_"+target_lr+".png")

        #We can also plot the general CE hotspots for each ligand-receptor cluster
        hn.pl.lr_cluster_ce_hotspot_plot(lr_df=self.clustered_expressed_LR_df,
        cell_cci_centrality=cell_cci_centrality,
        adata=self.dataset, fname='output/general_ce_hotspot_'+self.name+"_"+target_lr+".png")

    def preprocessing_for_gcn_model(self):
        print("Preprocessing for GCN model...")
        #Select all target genes
        self.target_all_gene_expr, self.used_gene_list = hn.pr.get_gene_expr(self.dataset, self.expressed_lr_df, self.complex_db)
        #Now we need to build our feature matrix of cell types
        self.cell_type_tensor, self.cell_type_names = hn.pr.get_continuous_cell_type_tensor(self.dataset, continuous_cell_type_slot="predicted_cell_type")
        #And the adjancancy matrix of our cell network
        self.adjancancy_matrix = hn.pr.adj_normalize(adj=self.filtered_ce_tensor, cell_type_tensor=self.cell_type_tensor,
                                                     only_between_cell_type=True)

    def train_gcn_model(self, gene):
        print(f"Training GCN model for gene {gene}")
        #Now we can train a GCN-model for predicting the gene expression of a specific gene
        #First, we need to select the target gene to predict
        target = hn.pr.get_one_case_expr(self.target_all_gene_expr, cases_list=self.used_gene_list,
                                         used_case_name=gene)
        sc.pl.spatial(dataset, color=[gene], cmap="Spectral_r", size=1.4, alpha=0.7, save=f"output/{gene}.png")
        #We can then train our model
        if torch.cuda.is_available():
            print("Started training using GPU...")
            trained_MGC_model = hn.pr.mgc_repeat_training(self.cell_type_tensor, self.adjancancy_matrix,
                                                                    target, device='gpu')
        else:
            print("Started training using CPU...")
            trained_MGC_model = hn.pr.mgc_repeat_training(self.cell_type_tensor, self.adjancancy_matrix,
                                                                    target)
        return trained_MGC_model



    def visualize_gcn_output(self, gene, model, lr_pair="all"):
        print("Visualizing GCN output...")
        MGC_model_only_type_list, \
        used_gene_list = hn.pr.load_model_list(self.cell_type_tensor, self.adjancancy_matrix, project_name=name+"_only_type",
                                               only_cell_type=True)
        MGC_model_type_GCN_list, \
        used_gene_list = hn.pr.load_model_list(self.cell_type_tensor, self.adjancancy_matrix, project_name=name+"_GCN")
        #Now that our model is trained, we can visualize the model results
        for model in MGC_model_type_GCN_list:
            #Let's plot the top 15 LR pairs
            ranked_LR_df = hn.pl.lr_rank_in_mgc(model, self.expressed_lr_df,
                                                plot_cluster=False, repeat_attention_scale=True,
                                                fname="output/LR_ranking_"+self.name+"_"+gene+".png")
            cluster_ranked_LR_df = hn.pl.lr_rank_in_mgc(model, self.clustered_expressed_LR_df,
                                                plot_cluster=True, cluster_col=True, repeat_attention_scale=True,
                                                fname="output/LR_ranking_clustered_"+self.name+"_"+gene+".png")
            #Now we can plot the cell-type level FCE network
            _ = hn.pl.fce_cell_type_network_plot(model, self.expressed_lr_df, self.cell_type_tensor, self.adjancancy_matrix,
                                                 self.cell_type_names, plot_lr=lr_pair, edge_thres=0.2,
                                                 palette=hn.brca_default_color_celltype,
                                                 fname="output/fce_cell_type_network_"+self.name+"_"+self.gene+"_"+lr_pair+".png")
            #Plot Delta E proportion per cell type
            delta_e = hn.pl.delta_e_proportion(model, self.cell_type_tensor, self.adjancancy_matrix,
                                               self.cell_type_names, palette = hn.brca_default_color_celltype,
                                               fname="output/delta_plot_"+self.name+"_"+gene+".png")



    def multitarget_training(self, genes_to_plot=[]):
        print("Training GCN for all genes...")
        if not path.isdir("_tmp_save_model/"+self.name+"_GCN"):
            print("Train model for all genes..")
            #We can model all target genes to get an idea of the genes which are more affected by CCC
            if torch.cuda.is_available():
                MGC_model_only_type_list, \
                MGC_model_type_GCN_list = hn.pr.mgc_training_for_multiple_targets(self.cell_type_tensor,
                                                self.adjancancy_matrix, self.target_all_gene_expr, device='gpu')
            else:
                MGC_model_only_type_list, \
                MGC_model_type_GCN_list = hn.pr.mgc_training_for_multiple_targets(self.cell_type_tensor,
                                                self.adjancancy_matrix, self.target_all_gene_expr)
        else:
            print("Models were already trained, loading trained models...")
            MGC_model_only_type_list, \
            used_gene_list = hn.pr.load_model_list(self.cell_type_tensor, self.adjancancy_matrix, project_name=name+"_only_type",
                                                   only_cell_type=True)
            MGC_model_type_GCN_list, \
            used_gene_list = hn.pr.load_model_list(self.cell_type_tensor, self.adjancancy_matrix, project_name=name+"_GCN")
        #Predict the expression using GCN+cell type vs only using cell type
        predicted_expr_type_GCN_df = hn.pr.get_mgc_result_for_multiple_targets(MGC_model_type_GCN_list,
                                                                        self.cell_type_tensor, self.adjancancy_matrix,
                                                                        self.used_gene_list, self.dataset)
        predicted_expr_only_type_df = hn.pr.get_mgc_result_for_multiple_targets(MGC_model_only_type_list,
                                                                        self.cell_type_tensor, self.adjancancy_matrix,
                                                                        self.used_gene_list, self.dataset)
        #We can compare the pearson correlation between the two predictions to identify CCC-dominated genes
        only_type_vs_GCN_all = hn.pl.find_genes_linked_to_ce(predicted_expr_type_GCN_df,
                                                             predicted_expr_only_type_df,
                                                             self.used_gene_list, self.target_all_gene_expr,
                                                             plot_gene_list=genes_to_plot, linewidths=5,
                                                             fname="output/pred_correlation_"+self.name)
        only_type_vs_GCN_all.to_csv("output/correlation_diff_df_"+name+".csv")

        #Save all results
        all_target_result = hn.pl.save_mgc_interpretation_for_all_target(MGC_model_type_GCN_list,
                                                                         self.cell_type_tensor, self.adjancancy_matrix,
                                                                 self.used_gene_list, self.expressed_lr_df.reset_index(), self.cell_type_names,
                                                                 LR_pair_num_per_target=15,
                                                                 heatmap_plot_lr_num=15,
                                                                 edge_thres=0.2,
                                                                 save_fce_plot=True,
                                                                 palette=hn.brca_default_color_celltype,
                                                                 figures_save_folder='./output/all_target/',
                                                                 project_name=name+"_all_targets")
        #Save the only-type models
        hn.pr.save_model_list(MGC_model_only_type_list, project_name=name+"_only_type",
                              target_gene_name_list=self.used_gene_list)
        #Save the GCN models
        hn.pr.save_model_list(MGC_model_type_GCN_list, project_name=name+"_GCN",
                              target_gene_name_list=self.used_gene_list)




print("args given: ", args)
print("Loading dataset...")
if args.dataset == 'brca_visium':
    #Load example Visium dataset (24,923 genes, 3798 spots)
    dataset = hn.pp.load_brca_visium_10x()
    name = 'brca_visium'
    organism = 'human'

elif args.dataset == 'resolve':
    if not os.path.exists(dirpath+"/data"):
        os.mkdir("data")
    if not os.path.exists(dirpath+"/data/resolve.h5ad"):
        print("Downloading RESOLVE dataset:")
        link = requests.get("https://dl01.irc.ugent.be/spatial/adata_objects/adataA1-1.h5ad")
        with open('data/resolve.h5ad', 'wb') as f:
            f.write(link.content)
    dataset = sc.read("data/resolve.h5ad")
    organism = 'mouse'

elif args.dataset == 'nanostring':
    dataset = sq.read.nanostring(path="data/Lung5_Rep1",
                       counts_file="Lung5_Rep1_exprMat_file.csv",
                       meta_file="Lung5_Rep1_metadata_file.csv",
                       fov_file="Lung5_Rep1_fov_positions_file.csv")
    organism = 'human'
    #Subset nanostring data in 4 parts
    print(dataset.obs.shape)


else:
    print("No dataset selected...")
    sys.exit()

#Make sure the plot layout works correctly
plt.rcParams.update({'figure.autolayout':True, 'savefig.bbox':'tight'})

print(f"Analyzing {dataset} from {organism}...")
holonet_pipeline(dataset, organism, list_of_target_lr=[], list_of_target_genes=[])
