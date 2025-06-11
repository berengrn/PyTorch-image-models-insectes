import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict

def Build_H_Matrix(data):

    # Calcule l'offset pour chaque niveau
    n_ordres = len(ordres)
    n_familles = len(familles)
    n_genres = len(genres)
    n_especes = len(especes)

    idx_ordres = {name: i for i, name in enumerate(ordres)}
    idx_familles = {name: i + n_ordres for i, name in enumerate(familles)}
    idx_genres = {name: i + n_ordres + n_familles for i, name in enumerate(genres)}
    idx_especes = {name: i + n_ordres + n_familles + n_genres for i, name in enumerate(especes)}

    # Fusion pour accès global
    idx_all = {**idx_ordres, **idx_familles, **idx_genres, **idx_especes}

    parent_to_children = defaultdict(set)

    for sp_name, info in data.items():
        order = info['order']
        family = info['family']
        genus = info['genus']
        species = info['species']

        # Ajoute les relations dans le graphe
        parent_to_children[idx_ordres[order]].add(idx_familles[family])
        parent_to_children[idx_familles[family]].add(idx_genres[genus])
        parent_to_children[idx_genres[genus]].add(idx_especes[species])

    # La structure de données parent_to_children permet de connaître la liste des noeuds fils de chaque noeud parent.
    parent_to_children = {k: list(v) for k, v in parent_to_children.items()}


    ### A COMPLETER
    def build_H_matrix(parent_to_children, total_nodes):
        N = n_ordres + n_familles + n_genres + n_especes
        H = np.zeros((N,N))

        for i in range(N):
            for j in parent_to_children[i]:
                H[i][j] = 1
        return H


    total_nodes = n_ordres + n_familles + n_genres + n_especes
    H = build_H_matrix(parent_to_children, total_nodes)

class Hierarchical_dep_loss(nn.Module):
    def __init__(self):
        super.__init__()
    
    def forward(self,y_pred,y_true):
        pass