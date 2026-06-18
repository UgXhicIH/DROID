# Authorship
__author__ = "Rodrigo Ochoa"
__email__ = "raoc@novonordisk.com"
__version__ = "Github version, corrected by Morgan Letoux"
__modification__ = "MorganFigerprint vector updated, T-Sne Modified, now use a densMAP, corrected an unused variable, erased the stockage" \
"of sims matrix (Tanimoto²), more colors, export to html for graph export"

#PepFunn
#from pepfunn.similarity import Alignment


# Sys functions
import matplotlib.pyplot as plt #type: ignore
from matplotlib.markers import MarkerStyle #type: ignore
import pandas as pd #type: ignore
import numpy as np #type: ignore
import math
import tempfile
import subprocess
import os
import sys
from statistics import median, mean
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# RDKit
import rdkit #type: ignore
from rdkit import Chem #type: ignore
from rdkit import DataStructs #type: ignore
from rdkit.ML.Cluster import Butina #type: ignore
from rdkit.Chem import AllChem #type: ignore 
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat, GetTanimotoDistMat #type: ignore
from rdkit.Chem import Descriptors, Descriptors3D, rdMolDescriptors #type: ignore
from rdkit.Chem import rdFingerprintGenerator #type: ignore
from rdkit.Chem.Draw import IPythonConsole #type: ignore
from rdkit import DataStructs #type: ignore

# Scikit
from sklearn.decomposition import PCA #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.cluster import KMeans #type: ignore
from sklearn.metrics import silhouette_samples, silhouette_score #type: ignore
from sklearnex import patch_sklearn #type: ignore
patch_sklearn()
import seaborn as sns #type: ignore

class simClustering:
    def __init__(self, ids=[], molfiles=[], sequences=[]):
        # List containing the metadata
        self.comps=[]
        self.ids=[]
        self.sequences=[]
        self.excluded=[] 

        if molfiles:
            for i,molfile in enumerate(molfiles):
                mol = rdkit.Chem.MolFromMolBlock(molfile)
                self.comps.append(mol)
                if ids:
                    self.ids.append(ids[i])
                else:
                    self.ids.append(f'mol{i+1}')
            
                if sequences:
                    self.sequences.append(sequences[i])
                else:
                    raise ValueError("You should provide a list with the sequences. Please correct")
                    sys.exit(1)
        else:
            if sequences:
                for j,seq in enumerate(sequences):
                    try:
                        helm = ".".join(list(seq))
                        mol = rdkit.Chem.MolFromHELM("PEPTIDE1{%s}$$$$" % helm)
                        if mol is not None:
                            self.sequences.append(seq)
                            self.comps.append(mol)
                            
                            if ids:
                                self.ids.append(ids[j])
                            else:
                                self.ids.append(f'mol{j+1}')
                    except:
                        self.excluded.append(seq)
            else:
                raise ValueError("You should provide sequences with or without molfiles. Please correct")
                sys.exit(1)

    def run_clustering(self, cutoff=0.3):
        # Get the fingerprints
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
        fps = [morgan_gen.GetFingerprint(x) for x in self.comps]
        # First generate the distance matrix
        self.dists = []
        nfps = len(fps)
        
        for i in range(0,nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            self.dists.extend([1.0-x for x in sims])
        
        # now cluster the data:
        self.clusters = rdkit.ML.Cluster.Butina.ClusterData(self.dists,nfps,cutoff,isDistData=True)

    def get_centroids(self):
        # Get the centroids for clusters larger than 1 element
        centroids={}
        for i,clus in enumerate(self.clusters):
            if len(clus)>1:
                centroids[self.sequences[clus[0]]]=len(clus)

        # Sort by cluster size and store in dictionary
        sort_centroids = dict(sorted(centroids.items(), key=lambda item: item[1], reverse=True))
        return sort_centroids

    def plot_clusters(self, centroids):
        labels=[]
        values=[]
        for j,key in enumerate(centroids):
            labels.append(f'cluster_{j+1}')
            values.append(centroids[key])

        # Create horizontal bar plot
        fig, ax = plt.subplots()
        ax.barh(labels, values)

        # Add labels to bars
        for i, v in enumerate(values):
            ax.text(v + 1, i, str(v), ha='center')

        # Set axis labels and title
        ax.set_xlabel('Number of elements')
        ax.set_ylabel('Sequences')
        ax.set_title('Clustering')

        # Show plot
        plt.show()
    
    def get_sim_reference(self, reference):

        # Select a molecule of interest and find its neighbors
        mol_idx = self.ids.index(reference)
        neighbors = self.ids
        mols = self.comps

        # Get the fingerprints
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
        fps = [morgan_gen.GetFingerprint(x) for x in mols]
        fps_ref = morgan_gen.GetFingerprint(self.comps[mol_idx])

        # First generate the distance matrix
        distances = []
        for i,fps_comp in enumerate(fps):
            sim = DataStructs.TanimotoSimilarity(fps_comp,fps_ref)
            distances.append(sim)
        
        # Get the distances
        neighbor_dist={}
        for j,n in enumerate(neighbors):
            neighbor_dist[n]=distances[j]
            
        sort_neighbors = dict(sorted(neighbor_dist.items(), key=lambda item: item[1], reverse=True))

        return sort_neighbors
    
    def get_sim_cluster(self, reference):

        # Select a molecule of interest and find its neighbors
        mol_idx = self.ids.index(reference)
        cluster_idx = [i for i, cluster in enumerate(self.clusters) if mol_idx in cluster][0]

        neighbors = [self.ids[idx] for idx in self.clusters[cluster_idx] if idx != mol_idx]

        mols = [self.comps[idx] for idx in self.clusters[cluster_idx] if idx != mol_idx]
        # Get the fingerprints
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
        fps = [morgan_gen.GetFingerprint(x) for x in mols]
        fps_ref = morgan_gen.GetFingerprint(self.comps[mol_idx])

        # First generate the distance matrix
        distances = []
        for i,fps_comp in enumerate(fps):
            sim = DataStructs.TanimotoSimilarity(fps_comp,fps_ref)
            distances.append(sim)

        # Get the distances
        neighbor_dist={}
        for j,n in enumerate(neighbors):
            neighbor_dist[n]=distances[j]
            
        sort_neighbors = dict(sorted(neighbor_dist.items(), key=lambda item: item[1], reverse=True))
        
        return sort_neighbors
    
    def compare_batch_reference(self, ref_list):

        # Iterate over the the reference list
        for i,ref in enumerate(ref_list):
            if i == 0:
                neighbours=self.get_sim_reference(ref)
                df_neighbours=pd.DataFrame.from_dict(neighbours, orient='index')
                df_neighbours = df_neighbours.rename(columns={0: ref})
            else:
                neighbours=self.get_sim_reference(ref)
                df_neighbours_new=pd.DataFrame.from_dict(neighbours, orient='index')
                df_neighbours_new = df_neighbours_new.rename(columns={0: ref})
                df_neighbours = df_neighbours.join(df_neighbours_new)
        
        return df_neighbours
    
    def compare_clusters(self, clus_id1, clus_id2):

        # Select the molecules from both clusters
        mols1 = [self.comps[idx] for idx in self.clusters[clus_id1]]
        mols2 = [self.comps[idx] for idx in self.clusters[clus_id2]]

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
        similarity_matrix = []
        for mol1 in mols1:
            row = []
            fp1 = morgan_gen.GetFingerprint(mol1)
            for mol2 in mols2:
                fp2 = morgan_gen.GetFingerprint(mol2)
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                row.append(similarity)
            similarity_matrix.append(row)

        # Calculate the average similarity between the two lists of molecules
        average_similarity = sum(max(row) for row in similarity_matrix) / len(similarity_matrix)
        
        return average_similarity
    
    def plot_sim_space_plotly(self, add_cluster_color=False, out_name='cluster_interactive.html'):
        import plotly.express as px # type: ignore
        import os
        import pandas as pd # type: ignore
        #from sklearn.manifold import TSNE # type: ignore
        import umap #type: ignore
        import numpy as np # type: ignore
        out_path = os.getcwd()

        # 1. Calcul des distances et t-SNE
        n_mol = len(self.comps)
        similarity_matrix = np.ones([n_mol,n_mol])
        i_lower= np.tril_indices(n=n_mol,m=n_mol,k=-1)
        i_upper= np.triu_indices(n=n_mol,m=n_mol,k=1)
        similarity_matrix[i_lower] = [1.0 - d for d in self.dists]
        similarity_matrix[i_upper] = similarity_matrix.T[i_upper]
        
        distance_matrix = np.subtract(1,similarity_matrix)
        
        #if n_mol<1: perp=1
        #elif n_mol<=50: perp= max(2, n_mol//3)
        #else: perp=(n_mol//100)

        UMAP_sim = umap.UMAP(n_neighbors=20, min_dist=0.1,n_components=3, densmap=True, dens_lambda=2.0, metric='precomputed').fit_transform(distance_matrix)
        UMAP_result = pd.DataFrame(data= UMAP_sim, columns=['UMAP1', 'UMAP2', 'UMAP3'])
        #TSNE_sim = TSNE(n_components=2,init='pca',random_state=90, angle = 0.3, perplexity=perp).fit_transform(distance_matrix)
        #tsne_result = pd.DataFrame(data = TSNE_sim , columns=["TC1","TC2"])
        #tsne_result.to_excel("T-SNE_Plot_PepFuNN.xlsx", index=False)
        
        # Ajout des métadonnées
        UMAP_result['Accession'] = self.ids
        UMAP_result['Sequence'] = self.sequences
        
        #tsne_result['Accession'] = self.ids
        #tsne_result['Sequence'] = self.sequences

        # Gestion des clusters
        cluster_labels=[]
        for idx in range(0, len(self.ids)):
            for i, cluster in enumerate(self.clusters):
                if idx in cluster:
                    if i < 35:
                        cluster_labels.append(str(i))
                    else:
                        cluster_labels.append('Others')
        
        UMAP_result['Cluster'] = cluster_labels
        # tsne_result['Cluster'] = cluster_labels

        # 3. Création du graphique interactif Plotly
        #For 3D UMAP
        if add_cluster_color:
            # Trie Legende
            UMAP_result = UMAP_result.sort_values(by="Cluster") 
            fig = px.scatter_3d(
                UMAP_result, 
                x="UMAP1", 
                y="UMAP2",
                z="UMAP3", 
                color="Cluster",
                hover_data={"Accession": True, "Sequence": True, "UMAP1": False, "UMAP2": False, 'UMAP3':False},
                title="Clustering par PepFuNN (UMAP 3D)",
                color_discrete_sequence=px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
            )
        else:
            fig = px.scatter_3d(
                UMAP_result, 
                x="UMAP1", 
                y="UMAP2",
                z="UMAP3", 
                color="Cluster",
                hover_data={"Accession": True, "Sequence": True, "UMAP1": False, "UMAP2": False, 'UMAP3':False},
                title="Clustering par PepFuNN (UMAP 3D)",
                color_discrete_sequence=px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
            )

        #For TSNE
        #if add_cluster_color:
            # Trie Legende
            #tsne_result = tsne_result.sort_values(by="Cluster") 
            #fig = px.scatter(
                #tsne_result, 
                #x="UMAP1", 
                #y="UMAP2", 
                #color="Cluster",
                #hover_data={"Accession": True, "Sequence": True, "UMAP1": False, "UMAP2": False},
                #title="Clustering par PepFuNN",
                #color_discrete_sequence=px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
            #)
        #else:
            #fig = px.scatter(
                #tsne_result, 
                #x="UMAP1", 
                #y="UMAP2",
                #hover_data={"Accession": True, "Sequence": True, "UMAP1": False, "UMAP2": False},
                #title="Clustering PepFuNN Interactif (t-SNE)"
            #)

        # Esthétique
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(template="simple_white", hovermode="closest", showlegend=True)

        # 4. Sauvegarde en fichier HTML
        cluster_name = os.path.join(out_path, out_name)
        fig.write_html(cluster_name)
        print(f"Graphique interactif sauvegardé sous : {cluster_name}")

        plt.show()
   
class propClustering:

    def __init__(self, ids=[], molfiles=[], sequences=[]):

        # List containing the metadata
        self.comps=[]
        self.ids=[]
        self.sequences=[]
        self.excluded=[] 

        if molfiles:
            for i,molfile in enumerate(molfiles):
                mol = rdkit.Chem.MolFromMolBlock(molfile)
                self.comps.append(mol)
                if ids:
                    self.ids.append(ids[i])
                else:
                    self.ids.append(f'mol{i+1}')
            
                if sequences:
                    self.sequences.append(sequences[i])
                else:
                    raise ValueError("You should provide a list with the sequences. Please correct")
                    sys.exit(1)
        else:
            if sequences:
                for j,seq in enumerate(sequences):
                    try:
                        helm = ".".join(list(seq))
                        mol = rdkit.Chem.MolFromHELM("PEPTIDE1{%s}$$$$" % helm)
                        if mol is not None:
                            self.sequences.append(seq)
                            self.comps.append(mol)
                            
                            if ids:
                                self.ids.append(ids[j])
                            else:
                                self.ids.append(f'mol{j+1}')
                    except:
                        self.excluded.append(seq)
            else:
                raise ValueError("You should provide sequences with or without molfiles. Please correct")
                sys.exit(1)

        # Calculate properties
        self.table=pd.DataFrame()
        data_list=[]
        for i,mol in enumerate(self.comps):
            rdkit.Chem.SanitizeMol(mol)
            row_data = {
                'smiles': rdkit.Chem.MolToSmiles(mol),
                'Mol': mol,
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
                'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'RingCount': Descriptors.RingCount(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'TPSA': Descriptors.TPSA(mol)
            }
            data_list.append(row_data)
        self.table = pd.DataFrame(data_list)
            
        
        self.desc_used=['MolWt','LogP','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds',
                        'NumHeavyAtoms','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings',
                        'NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','RingCount','FractionCSP3',
                        'TPSA']
    
    ########################################################################################
    def plot_PCA(self, reference=[], add_cluster=False, list_desc=[], color_index=[], print_arrows=False, legends=False, out_name='properties.png'):

        if not list_desc:
            list_desc=self.desc_used

        # Select a molecule of interest and find its neighbors
        if reference:
            mol_idx_list=[]
            for ref in reference:
                try:
                    mol_idx_list.append(self.ids.index(ref))
                except:
                    print(f'Reference molecule {ref} was not included in the property analysis.')

        # Path to save the image
        out_path = os.getcwd()

        descriptors = self.table[list_desc].values

        # Scaling descriptors
        descriptors_std = StandardScaler().fit_transform(descriptors)
        pca = PCA()
        descriptors_2d = pca.fit_transform(descriptors_std)
        
        # Saving PCA values to a new table
        descriptors_pca= pd.DataFrame(descriptors_2d)
        descriptors_pca.index = self.table.index
        descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns] #type: ignore

        # Plot PC1 and PC2
        scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
        scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

        # And we add the new values to our PCA table
        descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
        descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]

        if add_cluster:
            # K-means clustering
            kmeans = KMeans(init='k-means++', n_init=10, random_state=10)
            clusters = kmeans.fit(pd.DataFrame(descriptors_std))
            descriptors_pca['Cluster'] = pd.Series(clusters.labels_, index=descriptors_pca.index)

        plt.rcParams['axes.linewidth'] = 1.5
        plt.figure(figsize=(6,6))

        if add_cluster:
            ax=sns.scatterplot(x='PC1_normalized',y='PC2_normalized', data=descriptors_pca, hue='Cluster', s=20, palette=sns.color_palette("Set2", len(list(set(clusters.labels_)))), linewidth=0.2, alpha=1)
        else:
            ax=sns.scatterplot(x='PC1_normalized',y='PC2_normalized', data=descriptors_pca, s=20, color=sns.color_palette("Set2", 3)[0], linewidth=0.2, alpha=1)        
            if not legends:
                for index, row in descriptors_pca.iterrows():
                    for mol_idx in mol_idx_list:
                        if index==mol_idx:
                            plt.scatter(x=row['PC1_normalized'], y=row['PC2_normalized'], s=40, c='pink', marker=MarkerStyle('o'), edgecolors='black', linewidths=1)
            else:
                colors = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'pink', 'brown', 'gray', 'black', 'turquoise', 'magenta', 'lavender', 'teal', 'maroon', 'navy', 'olive', 'cyan', 'indigo']
                for index, row in descriptors_pca.iterrows():
                    for mol_idx in mol_idx_list:
                        if index==mol_idx:
                            if color_index:
                                plt.scatter(x=row['PC1_normalized'], y=row['PC2_normalized'], s=40, c=colors[color_index[counter]], marker=MarkerStyle('o'), edgecolors='black', linewidths=1, label=reference[counter], alpha=1)
                            else:
                                plt.scatter(x=row['PC1_normalized'], y=row['PC2_normalized'], s=40, c=colors[counter], marker=MarkerStyle('o'), edgecolors='black', linewidths=1, label=reference[counter], alpha=1)
                            counter+=1
                            counter+=1
                plt.legend(loc='best',frameon=False,prop={'size': 10})
        
        # Store the clusters
        self.clusters = descriptors_pca

        plt.xlabel ('PC1',fontsize=14,fontweight='bold')
        ax.xaxis.set_label_coords(0.98, 0.45)
        plt.ylabel ('PC2',fontsize=14,fontweight='bold')
        ax.yaxis.set_label_coords(0.45, 0.98)

        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ## Components eigenvectors (main features) for PC1 and PC2
        l=np.transpose(pca.components_[0:2, :]) 
        n=2
        if print_arrows:
            for i in range(n):
                plt.arrow(0, 0, l[i,0], l[i,1],color= 'k',alpha=0.5,linewidth=1.8,head_width=0.025)
                plt.text(l[i,0]*1.25, l[i,1]*1.25, list_desc[i], color = 'k',va = 'center', ha = 'center', fontsize=14)

        circle = plt.circle((0,0), 1, color='gray', fill=False, clip_on=True, linewidth=1.5, linestyle='--') #type: ignore

        plt.tick_params ('both',width=2, labelsize=14)

        if add_cluster:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])
            plt.legend(loc='best',frameon=False,prop={'size': 10}, ncol=2)

        ax.add_artist(circle)
        plt.xlim(-1.2,1.2)
        plt.ylim(-1.2,1.2)
        plt.tight_layout()

        prop_name = os.path.join(out_path, out_name)
        plt.savefig(prop_name)
        plt.show()
    
    def calc_distances(self, ref_list):
        """
        Function to calculate euclidean distance between two vectors

        :param ref_list: list with ids to compare

        :return dictionary with the calculated distances
        """

        descriptors = self.table[self.desc_used].values

        # Scaling descriptors
        descriptors_std = StandardScaler().fit_transform(descriptors)

        # Generate distance matrix
        n = len(descriptors_std)

        for i,ref in enumerate(ref_list):
            idx=self.ids.index(ref)
            dists = {}
            for j in range(0,n):
                # create two arrays
                a = np.array(descriptors_std[idx])
                b = np.array(descriptors_std[j])
                distance = np.linalg.norm(a - b)
                neigh=self.ids[j]
                dists[neigh]=distance
            
            if i == 0:
                df_neighbours=pd.DataFrame.from_dict(dists, orient='index')
                df_neighbours = df_neighbours.rename(columns={0: ref})
            else:
                df_neighbours_new=pd.DataFrame.from_dict(dists, orient='index')
                df_neighbours_new = df_neighbours_new.rename(columns={0: ref})
                df_neighbours = df_neighbours.join(df_neighbours_new)

        return df_neighbours

def sequence_clustering(ids=[], sequences=[], threshold=0.5, min_len_clus=5, add_isolated=True):

    if sequences:
        if not ids:
            for j,seq in enumerate(sequences):        
                ids.append(f'mol{j+1}')
    else:
        raise ValueError("You should provide a list of sequences. Please correct")
        sys.exit(1)
     
    clusters={}
    clus_ids={}
    mapped=[]
    mapped_ids=[]
    isolated=[]
    iso_ids=[]

    cluster_num=1
    for z,seq1 in enumerate(sequences):
        if ids[z] not in mapped_ids:
            clusters[cluster_num]=[]
            clus_ids[cluster_num]=[]
            
            map_temp=[]
            map_temp_ids=[]
            for j,seq2 in enumerate(sequences):
                if z!=j:
                    if len(seq1) >= len(seq2):
                        for i in range(0, len(seq1) - len(seq2) + 1):
                            fragment = seq1[i:i + len(seq2)]
                            m=Alignment.align_samelen_local(fragment, seq2) #type: ignore
                            if m > len(seq2) * threshold:
                                if seq2 not in mapped and ids[j] not in mapped_ids:
                                    clusters[cluster_num].append(seq2)
                                    map_temp.append(seq2)

                                    clus_ids[cluster_num].append(ids[j])
                                    map_temp_ids.append(ids[j])
                                    break
                    else:
                        for i in range(0, len(seq2) - len(seq1) + 1):
                            fragment = seq2[i:i + len(seq1)]
                            m=Alignment.align_samelen_local(fragment, seq1) #type: ignore
                            if m > len(seq1) * threshold:
                                if seq2 not in mapped and ids[j] not in mapped_ids:
                                    clusters[cluster_num].append(seq2)
                                    map_temp.append(seq2)

                                    clus_ids[cluster_num].append(ids[j])
                                    map_temp_ids.append(ids[j])
                                    break

            # Create the clusters based on the minimum length
            if len(clusters[cluster_num])>=min_len_clus:
                clusters[cluster_num].append(seq1)
                mapped.append(seq1)
                mapped=mapped+map_temp
                
                clus_ids[cluster_num].append(ids[z])
                mapped_ids.append(ids[z])
                mapped_ids=mapped_ids+map_temp_ids
                cluster_num+=1
            else:
                isolated.append(seq1)
                iso_ids.append(ids[z])

    # Complete the isolated molecules
    if add_isolated:      
        for j,seq1 in enumerate(isolated):
            close_clus={}
            for key in clusters:
                matches=[]
                for seq2 in clusters[key]:
                    if len(seq1) >= len(seq2):
                        for i in range(0, len(seq1) - len(seq2) + 1):
                            fragment = seq1[i:i + len(seq2)]
                            m=Alignment.align_samelen_local(fragment, seq2) #type: ignore                   
                    else:
                        for i in range(0, len(seq2) - len(seq1) + 1):
                            fragment = seq2[i:i + len(seq1)]
                            m=Alignment.align_samelen_local(fragment, seq1) #type: ignore
                    matches.append(m)
                if matches:
                    close_clus[key]=median(matches)
                else:
                    close_clus[key]=0
            
            
            sorted_dict = sorted(close_clus.items(), key=lambda x: x[1], reverse=True)
            selected_key = sorted_dict[0][0]
        
            clusters[selected_key].append(seq1)
            clus_ids[selected_key].append(iso_ids[j])
    
        dict_ids={'ID':[], 'cluster_id':[], 'sequence':[]}
        for key in clus_ids:
            for i,comp in enumerate(clus_ids[key]):
                dict_ids['ID'].append(comp)
                dict_ids['cluster_id'].append(key)
                dict_ids['sequence'].append(clusters[key][i])

        df_clus=pd.DataFrame(dict_ids)
    
    else:
        dict_ids={'ID':[], 'cluster_id':[], 'sequence':[]}
        for key in clus_ids:
            for i,comp in enumerate(clus_ids[key]):
                dict_ids['ID'].append(comp)
                dict_ids['cluster_id'].append(key)
                dict_ids['sequence'].append(clusters[key][i])
                
        for j,ele in enumerate(iso_ids):
            dict_ids['ID'].append(ele)
            dict_ids['cluster_id'].append('isolated')
            dict_ids['sequence'].append(isolated[j])

        df_clus=pd.DataFrame(dict_ids)

    return clusters, clus_ids, isolated, iso_ids, df_clus