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

    parent_to_children = {k: list(v) for k, v in parent_to_children.items()}

    N = n_ordres + n_familles + n_genres + n_especes
    H = np.zeros((N,N))

    for i in range(N):
        for j in parent_to_children[i]:
            H[i][j] = 1
    return H

class TaxaNet_custom_loss(nn.Module):

    def __init__(self,data):
        super.__init__()
        self.data = data
    
    def forward(self,y_pred,y_true):
        H = Build_H_Matrix(self.data)
        loss = 0.0
        sum = 0.0
        weights = [0.25, 0.25, 0.15, 0.1] #Les poids sont sans doute à re-tester pour notre dataset (pas les mêmes niveaux taxonomiques)
        levels_pred = torch.unbind(y_pred,axis = 1)
        levels_true = torch.unbind(y_true,axis = 1)
        for j in range(1,len(y_pred[0])):
            sum = 0.0
        #on itère d'abord le long des colonnes (prédiction d'un même niveau taxonomique pour chaque éléments du dataset)
            for i in range(len(y_pred)):
                sum += ((H[self.data[y_pred[i][j]]][self.data[y_pred[i][j-1]]] == 1)*np.e)
            sum += nn.CrossEntropyLoss(levels_pred[j],levels_true[j])
            sum *= weights[j]
            loss += sum
        return sum