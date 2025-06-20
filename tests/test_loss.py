import torch
import torch.nn as nn
import numpy as np
import os

from pathlib import Path

# Ajoute le dossier racine du projet (2 niveaux au-dessus du script)
#sys.path.append(str(Path(__file__).resolve().parents[2]))

from timm.loss import TaxaNetLoss,BuildDictionaries
from timm.loss.taxanet_custom_loss import Build_H_Matrix

hierarchy_csv = os.path.join(os.getcwd(),"hierarchy_test.csv")

y_true = torch.tensor([[0,2,5,9],[0,2,6,10],[0,3,7,11]])

y_correct = torch.tensor([[1.,0.,1.,0.,0.,1.,0.,0.,0.,1.,0.,0.], #résultat parfait
                          [1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.],  
                          [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.]
                        ])

y_false1 = torch.tensor([[1.,0.,0.,0.,1.,1.,0.,0.,0.,1.,0.,0.], #graves erreurs hiérarchiques
                         [1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,1.,0.],  
                         [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.]
                        ])

y_false2 = torch.tensor([[1.,0.,1.,0.,0.,1.,0.,0.,0.,0.,0.,1.], #légères erreurs hiérarchiques
                         [1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.],  
                         [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.]
                        ])

y_false3 = torch.tensor([[1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.], #erreurs de prédiction, mais respectant la hiérarchie
                         [1.,0.,1.,0.,0.,1.,0.,0.,0.,1.,0.,0.],  
                         [1.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.]
                        ])

loss_fn = TaxaNetLoss(hierarchy_csv)

print("loss pour une prédiction avec de grosses erreurs hiérachiques")
print(loss_fn(y_false1,y_true))

print("loss pour une prédiction avec de légères erreurs hiérachiques")
print(loss_fn(y_false2,y_true))

print("loss pour une prédiction avec des erreurs de prédiction mais pas d'erreurs hiérarchiques")
print(loss_fn(y_false3,y_true))

print("loss pour un résultat exact")
print(loss_fn(y_correct,y_true))
