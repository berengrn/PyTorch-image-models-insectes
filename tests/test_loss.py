import torch
import torch.nn as nn
import numpy as np
import os

from pathlib import Path

# Ajoute le dossier racine du projet (2 niveaux au-dessus du script)
#sys.path.append(str(Path(__file__).resolve().parents[2]))

from timm.loss import TaxaNetLoss,BuildDictionaries
from timm.loss.taxanet_custom_loss import Build_H_Matrix

hierarchy_csv = os.path.join(os.getcwd(),"hierarchy-test.csv")
#niveau 1: 2 classes (0,1), niveau 2: 3 classes (2,3,4), niveau 3: 4 classes (5,6,7,9) (total:9)

parent_to_children,dict_especes,N = BuildDictionaries(hierarchy_csv)

y_true = torch.tensor([[[1.,0.,1.,0.,0.,1.,0.,0.,0.,0.],
                        [1.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                        [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.],
                        ]])

y_false_1 = torch.tensor([[[0.,1.,1.,0.,0.,1.,0.,0.,0.,0.], #grosses erreurs hiérachiques
                        [0.,1.,1.,0.,0.,0.,1.,0.,0.,0.],
                        [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.],
                        ]])

y_false_2 = torch.tensor([[[1.,0.,1.,0.,0.,0.,0.,0.,0.,1.], #petites erreurs hiérarchiques
                        [1.,0.,1.,0.,0.,0.,0.,0.,0.,1.],
                        [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.],
                        ]])

y_false_3 = torch.tensor([[[1.,0.,1.,0.,0.,0.,1.,0.,0.,0.],  #erreurs de prédictions mais hiérachie respectée
                        [1.,0.,1.,0.,0.,1.,0.,0.,0.,0.],
                        [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.],
                        ]])

loss_fn = TaxaNetLoss(hierarchy_csv)

print("loss pour une prédiction avec de grosses erreurs hiérachiques")
print(loss_fn(y_false1,y_true))

print("loss pour une prédiction avec de légères erreurs hiérachiques")
print(loss_fn(y_false2,y_true))

print("loss pour une prédiction avec des erreurs de p^rédiction mais pas d'erreurs hiérarchiques")
print(loss_fn(y_false3,y_true))

print("loss pour un résultat exact")
print(loss_fn(y_true,y_true))
