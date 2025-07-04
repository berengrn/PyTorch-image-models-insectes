#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os
import csv

from collections import defaultdict

def BuildDictionaries(csv_path):

    ordres = []
    familles = []
    genres = []
    especes = []
    data = {}

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            espece = row['species']
            ordre = row['order']
            famille = row['family']
            genre = row['genus']

            if ordre not in ordres:
                ordres.append(ordre)
            if famille not in familles:
                familles.append(famille)
            if genre not in genres:
                genres.append(genre)
            if espece not in especes:
                especes.append(espece)

            data[espece] = {
                'order': ordre,
                'family': famille,
                'genus': genre,
                'species': espece
            }

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

    #les piquets indiquent ou s'arrête chaque partie du vecteur correspondant à une niveau hiérarchique
    piquets = torch.cumsum(torch.tensor([0,n_ordres,n_familles,n_genres,n_especes]),0)

    return (parent_to_children,idx_all,piquets)

def Build_H_Matrix(parent_to_children,piquets):
    #Construit H telle que H[i][j] == 1 ssi i est parent de j, 0 sinon
    N = piquets[-1]
    H = np.zeros((N,N))
    for i in range(N):
        if i in parent_to_children:
            for j in parent_to_children[i]:
                H[i][j] = 1
    return H

class TaxaNetLoss(nn.Module):

    def __init__(self,csv_path):
        super().__init__()
        parent_to_children,_,self.piquets = BuildDictionaries(csv_path)
        self.H = Build_H_Matrix(parent_to_children,self.piquets)
    
    def forward(self,y_pred,y_true):
        loss = 0.0
        sum = 0.0
        weights = [0.25, 0.25, 0.15, 0.1] #Les poids sont sans doute à re-tester pour notre dataset (pas les mêmes niveaux taxonomiques)
        N,C_total = y_pred.size()
        nbLevels = len(self.piquets) - 1

        """
        N: Number of elements in the dataset
        C_total: Number of classes, all hierarchical levels combined
        nbLevels: Number of levels in the hierarchy
        """

        #arrays indiquant quels indices annuler pour ne garder que les classes k-ième niveau de la hiérarchie
        levels_supports = [torch.tensor([int(i >= self.piquets[k -1] and i < self.piquets[k]) for i in range(C_total)]) for k in range(1,nbLevels+1)]

        #tenseurs sur lesquels appliquer la cross-entropy loss:
        levels_pred = torch.stack([ torch.stack([y_pred[j] * levels_supports[i] for j in range(N)])  for i in range(nbLevels)])

        levels_true = torch.unbind(y_true,dim = 1)

        
        ce_fn = nn.CrossEntropyLoss()
        for k in range(1,nbLevels):
        #on itère d'abord sur chaque niveau hiérarchique
            sum = 0.0

            for i in range(1,N):
                sum += (self.H[torch.argmax(levels_pred[k-1][i])][torch.argmax(levels_pred[k][i])] == 0)*np.e
            sum += ce_fn(levels_pred[k],levels_true[k]) 
            sum *= weights[k]
            loss += sum

        return loss

if __name__ == '__main__':
    
    csv_path = os.path.join(os.pardir,"small-collomboles","hierarchy.csv")
    loss_fn = TaxaNetLoss(csv_path)
