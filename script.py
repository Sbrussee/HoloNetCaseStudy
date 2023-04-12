"""
Apply HoloNet on Liver data
2 Main experiments:
Normal vs. Disease sample
Parameter study (w_best, edge threshold)
"""

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
import anndata as ad
import matplotlib.pyplot as plt
import torch
import tiledb
import tiledbsoma


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
parser.add_argument('-hn', '--holonet', action='store_true', help='Whether to apply HoloNet', default=True)
parser.add_argument('-g', '--genes', help='List of target genes to query', default=[])
parser.add_argument('-p', '--pairs', help='List of ligand receptor pairs to query', default=[])
parser.add_argument('-a', '--all', action='store_true', help='Whether to plot all target genes')
parser.add_argument('-t', '--top', type=int, help='Whether to use the top x genes most influenced by FCEs as genes to query', default=None)
parser.add_argument('-v', '--visualize', action='store_true', help="Whether to plot the data spatially", default=False)
parser.add_argument('-c', '--cluster', action='store_true', help='Whether to cluster LR-pairs and visualize them.', default=False)
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

        if args.visualize == True:
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
        if len(list_of_target_lr) < 1:
            list_of_target_lr = self.expressed_lr_df['LR_Pair'].to_list()
        for pair in list_of_target_lr:
            self.visualize_ce_tensors(target_lr=pair)

        self.preprocessing_for_gcn_model()
        model_per_gene = {}
        if len(list_of_target_genes) < 1:
            list_of_target_genes = self.used_gene_list
        self.multitarget_training(list_of_target_genes, list_of_target_lr)



    def visualize_dataset(self):
        print("Visualizing dataset...")
        for cell_type in self.dataset.obs['cell_type']:
            #Plot cell type percentages
            hn.pl.plot_cell_type_proportion(self.dataset, plot_cell_type=cell_type,
                                            fname=f"cell_type_{cell_type}_proportions_{self.name}.png",
                                            spot_size=1)

        #Cell type labels per spot
        sc.pl.spatial(self.dataset, color=['cell_type'], size=1.4, alpha=0.7,
        palette=hn.brca_default_color_celltype, save=f"spatial_{self.name}.png", spot_size=1)



    def load_lr_df(self):
        print("Load expression matrix...")
        #Load human L-R database from CellChatDB, Connectome also possible
        self.interaction_db, self.cofactor_db, self.complex_db = hn.pp.load_lr_df(human_or_mouse=self.organism)
        #Filter LR-pairs by occuring at a percentage of cells (0.3)
        if path.exists("output/expressed_lr_df_"+self.name+".csv"):
            self.expressed_lr_df = pd.read_csv("output/expressed_lr_df_" + self.name + ".csv")
        else:
            self.expressed_lr_df = hn.pp.get_expressed_lr_df(self.interaction_db, self.complex_db,
                                                             self.dataset, expressed_prop=0.1)
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
        if args.visualize == True:
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
        self.ce_tensor = hn.tl.compute_ce_tensor(self.expressed_lr_df, self.w_best,
                                                 self.elements_expr_df_dict, self.dataset)
        #We can then filter the edges with low specifities
        if path.exists("output/ce_tensor_"+self.name+".pkl"):
            with open("output/ce_tensor_"+self.name+".pkl", 'rb') as f:
                self.tensor = pickle.load(f)
        else:
            hn.tl.filter_ce_tensor(self.ce_tensor, self.dataset, self.expressed_lr_df,
                                    self.elements_expr_df_dict, self.w_best, n_pairs=50, copy=False)
            with open("output/ce_tensor_"+self.name+".pkl", 'wb') as f:
                pickle.dump(self.ce_tensor, f)

    def visualize_ce_tensors(self, target_lr=''):
        print("Visualizing CE tensors...")
        #Now that we have our views, we can visualize each CE both on cell-level as well as on cell-type-level
        #We can use either degree or eigenvector centrality as CE strength per cell/spot
        #For example, let's see it for TGFB1:(TGFBR1+TGFBR2)
        if args.visualize == True:
            hn.pl.ce_hotspot_plot(self.ce_tensor, self.dataset,
            lr_df=self.expressed_lr_df, plot_lr=target_lr, fname='ce_hotspot_'+self.name+"_"+target_lr+".png")

            #Now based on eigenvector centrality
            hn.pl.ce_hotspot_plot(self.ce_tensor, self.dataset,
            lr_df=self.expressed_lr_df, plot_lr=target_lr, fname='ce_hotspot_eigenvector_'+self.name+"_"+target_lr+".png",
            centrality_measure='eigenvector')

        #We can also plot the cell-type CE network.
        #for this, we need to load the cell-type percentages per spot
        self.cell_type_mat, self.cell_type_names = hn.pr.get_continuous_cell_type_tensor(self.dataset,
                                                                              continuous_cell_type_slot='predicted_cell_type')

        _ = hn.pl.ce_cell_type_network_plot(self.ce_tensor, self.cell_type_mat, self.cell_type_names,
        lr_df=self.expressed_lr_df, plot_lr=target_lr, edge_thres=0.2,
        fname='cell_type_network_'+self.name+"_"+target_lr+".png")

        if args.cluster:
            #We can perform agglomerative clustering for the igand-receptor pairs based on the centrality measures.
            cell_cci_centrality = hn.tl.compute_ce_network_eigenvector_centrality(self.ce_tensor)
            self.clustered_expressed_LR_df, _ = hn.tl.cluster_lr_based_on_ce(self.ce_tensor, self.dataset, self.expressed_lr_df,
            w_best=self.w_best, cell_cci_centrality=cell_cci_centrality)

            #Now plot a dendogram using this clustering
            hn.pl.lr_clustering_dendrogram(_, self.expressed_lr_df, [target_lr],
            dflt_col="#333333",fname="clust_dendogram_"+self.name+"_"+target_lr+".png")

            #We can also plot the general CE hotspots for each ligand-receptor cluster
            hn.pl.lr_cluster_ce_hotspot_plot(lr_df=self.clustered_expressed_LR_df,
            cell_cci_centrality=cell_cci_centrality,
            adata=self.dataset, fname='general_ce_hotspot_'+self.name+"_"+target_lr+".png")

    def preprocessing_for_gcn_model(self):
        print("Preprocessing for GCN model...")
        #Select all target genes
        self.target_all_gene_expr, self.used_gene_list = hn.pr.get_gene_expr(self.dataset, self.expressed_lr_df, self.complex_db)
        #Now we need to build our feature matrix of cell types
        self.cell_type_tensor, self.cell_type_names = hn.pr.get_continuous_cell_type_tensor(self.dataset, continuous_cell_type_slot="predicted_cell_type")
        #And the adjancancy matrix of our cell network
        self.adjancancy_matrix = hn.pr.adj_normalize(adj=self.ce_tensor, cell_type_tensor=self.cell_type_tensor,
                                                     only_between_cell_type=True)

    def train_gcn_model(self, gene):
        print(f"Training GCN model for gene {gene}")
        #Now we can train a GCN-model for predicting the gene expression of a specific gene
        #First, we need to select the target gene to predict
        target = hn.pr.get_one_case_expr(self.target_all_gene_expr, cases_list=self.used_gene_list,
                                         used_case_name=gene)
        if args.visualize:
            sc.pl.spatial(self.dataset, color=[gene], cmap="Spectral_r", size=1.4, alpha=0.7, save=f"{gene}_truthexpr.png")
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
                                                fname="output/LR_ranking_"+self.name+"_trained_for_"+gene+".png")
            cluster_ranked_LR_df = hn.pl.lr_rank_in_mgc(model, self.clustered_expressed_LR_df,
                                                plot_cluster=True, cluster_col=True, repeat_attention_scale=True,
                                                fname="output/LR_ranking_clustered_"+self.name+"_trained_for_"+gene+".png")
            #Now we can plot the cell-type level FCE network
            _ = hn.pl.fce_cell_type_network_plot(model, self.expressed_lr_df, self.cell_type_tensor, self.adjancancy_matrix,
                                                 self.cell_type_names, plot_lr=lr_pair, edge_thres=0.2,
                                                 palette=hn.brca_default_color_celltype,
                                                 fname="output/fce_cell_type_network_"+self.name+"_trained_on_"+self.gene+"_"+lr_pair+".png")
            #Plot Delta E proportion per cell type
            delta_e = hn.pl.delta_e_proportion(model, self.cell_type_tensor, self.adjancancy_matrix,
                                               self.cell_type_names, palette = hn.brca_default_color_celltype,
                                               fname="output/delta_plot_"+self.name+"_trained_on_"+gene+".png")



    def multitarget_training(self, genes_to_plot=[], lr_to_plot=[]):
        print("Training GCN for all genes...")
        print(dirpath+"_tmp_save_model/"+self.name+"_GCN")
        if not path.isdir(os.path.join(dirpath, "_tmp_save_model/"+self.name+"_GCN")):
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
        predicted_expr_type_GCN_df.to_csv("output/pred_expr_GCN.csv")
        predicted_expr_only_type_df.to_csv("output/pred_expr_noGCN.csv")


        #We can compare the pearson correlation between the two predictions to identify CCC-dominated genes
        for gene in genes_to_plot:
            only_type_vs_GCN_all = hn.pl.find_genes_linked_to_ce(predicted_expr_type_GCN_df,
                                                                 predicted_expr_only_type_df,
                                                                 self.used_gene_list, self.target_all_gene_expr,
                                                                 plot_gene_list=[gene], linewidths=5,
                                                                 fname="output/pred_correlation_"+gene+"_"+self.name+".png")
            only_type_vs_GCN_all.to_csv("output/correlation_diff_df_"+name+".csv")
            top_genes = only_type_vs_GCN_all.head(args.top).index.values.tolist()

        if args.top != None:
            self.list_of_target_genes = top10_genes

        correlation_per_gene = {}
        #Lets visualize each model
        for model_list, gene in zip(MGC_model_type_GCN_list, self.list_of_target_genes):
            #Plot the predicted and true expression per gene
            plot = hn.pl.plot_mgc_result(model_list, self.dataset, self.cell_type_tensor, self.adjancancy_matrix,
                                              fname=f"pred_expr_GCN_{gene}_{self.name}")
            sc.pl.spatial(self.dataset, color=[gene], cmap="Spectral_r", size=1.4, alpha=0.7,
                          save=f'true_expr_{gene}_{self.name}')
            truth = hn.pr.get_one_case_expr(self.target_all_gene_expr, cases_list=self.used_gene_list,
                                            used_case_name=gene)
            predicted = hn.pr.get_mgc_result(model_list, self.cell_type_tensor, self.adjancancy_matrix)
            correlation_per_gene[gene] = np.corrcoef(predicted.T, truth.T)[0,1]

            #Plot the LR-ranking based on attention
            ranked_LR = hn.pl.lr_rank_in_mgc(model_list, self.expressed_lr_df,
                                             plot_cluster=False, repeat_attention_scale=True,
                                             fname=f"figures/ranked_LR_test_"+self.name+"_trained_on_"+gene+".png")
            _ = hn.pl.fce_cell_type_network_plot(model_list, self.expressed_lr_df, self.cell_type_tensor, self.adjancancy_matrix,
                                                 self.cell_type_names, plot_lr='all', edge_thres=0.2,
                                                 palette=hn.brca_default_color_celltype,
                                                 fname="figures/fce_cell_type_network_all"+self.name+"_trained_on_"+gene+".png")
            delta_e = hn.pl.delta_e_proportion(model_list, self.cell_type_tensor, self.adjancancy_matrix,
                                               self.cell_type_names, palette = hn.brca_default_color_celltype,
                                               fname="figures/delta_plot_"+self.name+"_trained_on_"+gene+".png")
            for pair in lr_to_plot:
                _ = hn.pl.fce_cell_type_network_plot(model_list, self.expressed_lr_df, self.cell_type_tensor, self.adjancancy_matrix,
                                                     self.cell_type_names, plot_lr=pair, edge_thres=0.2,
                                                     palette=hn.brca_default_color_celltype,
                                                     fname="figures/fce_cell_type_network_"+pair+"_"+self.name+"_trained_on_"+gene+".png")
        with open("output/truth_correlation_"+self.name, 'wb') as f:
            pickle.dump(correlation_per_gene, f)


        if args.all:
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

        #Visualize each model

        #Save the only-type models
        hn.pr.save_model_list(MGC_model_only_type_list, project_name=name+"_only_type",
                              target_gene_name_list=self.used_gene_list)
        #Save the GCN models
        hn.pr.save_model_list(MGC_model_type_GCN_list, project_name=name+"_GCN",
                              target_gene_name_list=self.used_gene_list)


def detect_hvgs(dataset):
    sc.pp.highly_variable_genes(dataset)
    return dataset


def lr_permutation_test(dataset, name):
    sc.pp.neighbors(dataset, copy=False)
    sc.tl.leiden(dataset, copy=False)
    sq.gr.ligrec(dataset, cluster_key='leiden',
                 interactions=None, copy=False, use_raw=False)
    #print(dataset.uns['leiden_ligrec']['means'])
    #print(dataset.uns['leiden_ligrec']['pvalues'])
    #sq.pl.ligrec(dataset, cluster_key='leiden')
    #plt.savefig(f"Ligand_Receptor_perm_{name}")
    #plt.close()
    return dataset

#Make sure the plot layout works correctly
plt.rcParams.update({'figure.autolayout':True, 'savefig.bbox':'tight'})


print("args given: ", args)
print("Loading dataset...")
if args.dataset == 'brca_visium':
    #Load example Visium dataset (24,923 genes, 3798 spots)
    dataset = hn.pp.load_brca_visium_10x()
    print(dataset.obsm['predicted_cell_type'])
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
    organism = 'human'
    full = sc.read("/srv/scratch/chananchidas/LiverData/LiverData_RawNorm.h5ad")
    #Subset nanostring data in 4 parts
    size_obs = full.X.shape[0]
    print(f'full size {size_obs}')
    #Split by tissue type
    normal, cancer = (full[full.obs['Run_Tissue_name'] == 'NormalLiver'],
                       full[full.obs['Run_Tissue_name'] == 'CancerousLiver'])
    print(normal, cancer)
    #Delete the full dataset from memory
    del full
    #Pass each through holonet
    for dataset in [normal, cancer]:
        fovs = np.unique(dataset.obs['fov'])
        tissue = str(dataset.obs['Run_Tissue_name'].unique()[0])
        for i in range(0,len(fovs)-10,10):
            fov = dataset[dataset.obs['fov'].isin(fovs[i:i+10])]
            fov = detect_hvgs(fov)
            fov = lr_permutation_test(fov, name="Nanostring_"+tissue+str(i)+str(i+10))
            for column in fov.uns.columns:
                print(column)
            fov.obs['cell_type'] = fov.obs['cellType']
            fov.obsm['X_spatial'], fov.obsm['Y_spatial'] = fov.obs['x_slide_mm'].to_frame(), fov.obs['y_slide_mm'].to_frame()
            fov.obsm['spatial'] = pd.concat([fov.obs['x_slide_mm'], fov.obs['y_slide_mm']], axis=1)
            predicted_cell_type = pd.get_dummies(fov.obs['cell_type']).apply(pd.Series.explode)
            max_cell_type = predicted_cell_type.idxmax(axis=1)
            fov.obsm['predicted_cell_type'] = pd.concat([predicted_cell_type, max_cell_type.rename('max')], axis=1)
            print(f"Saving {tissue} fov {i} to {i+10}...")
            print(fov.shape)
            del fov.raw
            #Save this sub-dataset
            fov.write(f'data/ns_fov_{tissue}_{i}_to_{i+10}.h5ad')
    for dataset in [f for f in os.listdir("data/") if f.startswith("ns_fov_")]:
        data = sc.read("data/"+dataset)
        tissue = str(data.obs['Run_Tissue_name'].unique()[0])
        i = np.min(data.obs['fov'])
        print(f"Analyzing {tissue} fov {i} to {i+10}...")
        holonet_pipeline(data, organism, name="Nanostring_"+tissue+str(i)+str(i+10),
        list_of_target_lr=args.pairs, list_of_target_genes=args.genes)



else:
    print("No dataset selected...")
    sys.exit()

#Make sure the plot layout works correctly
plt.rcParams.update({'figure.autolayout':True, 'savefig.bbox':'tight'})

print(f"Analyzing {dataset} from {organism}...")
list_of_target_lr = args.pairs.split(',')
list_of_target_genes = args.genes.split(',')
holonet_pipeline(dataset, organism, name=name,
                 list_of_target_lr=list_of_target_lr, list_of_target_genes=list_of_target_genes)
