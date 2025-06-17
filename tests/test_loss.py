 
import numpy as np
import os

from pathlib import Path

# Ajoute le dossier racine du projet (2 niveaux au-dessus du script)
#sys.path.append(str(Path(__file__).resolve().parents[2]))

from timm.loss import TaxaNet_custom_loss,Build_dictionnaries

hierarchy_csv = os.path.join(os.getcwd(),"..","small-collomboles","dataset","train")

_,dict_especes,_ = Build_dictionnaries(hierarchy_csv)

y_true_names = np.array([['Symphypleona','Sminthuridae','Allacma','Allacma fusca'],
                         ['Poduromorpha','Neanuridae','Anurida','Anurida maritima'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella aurantiaca'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella braunerae'],
                         ['Symphypleona','Bourletiellidae','Bourletiella','Bourletiella arvalis']
                         ])

#grosses erreurs dans la hiérarchie
y_false1_names = np.array([['Symphypleona','Sminthuridae','Anurida','Anurida maritima'],
                         ['Poduromorpha','Neanuridae', 'Allacma','Allacma fusca'],
                         ['Symphypleona','Bourletiellidae','Bourletiella','Bourletiella arvalis'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella braunerae'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella aurantiaca']
                         ])

#erreurs légères dans la hiérarchie
y_false2_names = np.array([['Symphypleona','Sminthuridae','Allacma','Allacma fusca'],
                         ['Poduromorpha','Neanuridae','Anurida','Anurida maritima'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella braunerae'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella aurantiaca'],
                         ['Symphypleona','Bourletiellidae','Bourletiella','Bourletiella hortensis']
                         ])

#erreurs, mais pas dans la hiérarchie
y_false3_names = np.array([
                         ['Poduromorpha','Neanuridae','Anurida','Anurida maritima'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella braunerae'],
                         ['Poduromorpha','Neanuridae','Bilobella','Bilobella aurantiaca'],
                         ['Symphypleona','Bourletiellidae','Bourletiella','Bourletiella hortensis'],
                         ['Symphypleona','Sminthuridae','Allacma','Allacma fusca']
                         ])

vectorized_map = np.vectorize(dict_especes.get)
y_false1 = vectorized_map(y_false1_names)
y_false2 = vectorized_map(y_false2_names)
y_false3 = vectorized_map(y_false3_names)
y_true = vectorized_map(y_true_names)

loss_fn = TaxaNet_custom_loss(hierarchy_csv)

print("loss pour une prédiction avec de grosses erreurs hiérachiques")
print(loss_fn(y_false1,y_true))

print("loss pour une prédiction avec de légères erreurs hiérachiques")
print(loss_fn(y_false2,y_true))

print("loss pour une prédiction avec des erreurs de p^rédiction mais pas d'erreurs hiérarchiques")
print(loss_fn(y_false3,y_true))

print("loss pour un résultat exact")
print(loss_fn(y_true,y_true))


