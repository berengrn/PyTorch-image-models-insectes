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

    return (parent_to_children,idx_all,(n_ordres + n_familles + n_genres + n_especes))

def Build_H_Matrix(parent_to_children,N):
    #H[i][j] == 1 signifie i parent de j
    H = np.zeros((N,N))

    for i in range(N):
        if i in parent_to_children:
            for j in parent_to_children[i]:
                H[i][j] = 1
    return H

class TaxaNetLoss(nn.Module):

    def __init__(self,csv_path):
        super().__init__()
        parent_to_children,_,N = BuildDictionaries(csv_path)
        self.H = Build_H_Matrix(parent_to_children,N)
    
    def forward(self,y_pred,y_true):
        loss = 0.0
        sum = 0.0
        weights = [0.25, 0.25, 0.15, 0.1] #Les poids sont sans doute à re-tester pour notre dataset (pas les mêmes niveaux taxonomiques)
        levels_pred = torch.unbind(y_pred,dim = 1)
        levels_true = torch.unbind(y_true,dim = 1)
        ce_fn = nn.CrossEntropyLoss()
        for j in range(1,len(y_pred[0])):
        #on itère d'abord le long des colonnes (prédictions pour les ordres, puis les familles, etc.)
            sum = 0.0
            for i in range(len(y_pred)):
                sum += ((self.H[round(y_pred[i,j-1].item())][round(y_pred[i,j].item())] != 1)*np.e) #H[i][j] == 1 signifie i parent de j
            sum += ce_fn(levels_pred[j],levels_true[j])
            print(ce_fn(levels_pred[j],levels_true[j]))
            sum *= weights[j]
            loss += sum
        return loss

if __name__ == '__main__':
    csv_path = os.path.join(os.pardir,"small-collomboles","hierarchy.csv")
    loss_fn = TaxaNetLoss(csv_path)