import csv
import os
from typing import List

import numpy as np
import pandas


def read_csv_as_dict(filepath, orient="list"):
    """Lis un fichier csv et le retourne sous forme de dictionnaire.

    Parameters
    ----------
    filepath : String
        Un lien relatif ou absolu vers le fichier que l'on veut lire.

    Returns
    -------
        Un dictionnaire ou les clés sont les indices du csv et les valeurs les valeurs associées.
    """
    return pandas.read_csv(filepath).to_dict(orient=orient)


def read_csv_as_xy(filepath, index_x: List, index_y: List = None):
    """Lis un fichier et le retourne sus la forme de deux listes qui représentent les données et l'étiquettage.

    Parameters
    ----------
    filepath : String
        Un lien relatif ou absolu vers le fichier que l'on veut lire.
    index_x : List[String]
        La liste des indices que l'on veut mettre dans la liste des données.  ex:('Bouton','Capteur'...).
    index_y : List[String]
        La liste des indices que l'on veut mettre dans la liste d'étiquettage. Typiquement 'Bouton'.

    Returns
    -------
        Un tuple contenant les liste de données et d'étiquetage.
    """
    if index_y is None:
        index_y = []
    dataframe = pandas.read_csv(filepath)
    x = np.array(dataframe[index_x]).T
    y = np.array(dataframe[index_y]).T
    return x, y


def get_files_from_dir(dirpath):
    """Liste tout les fichiers dans un dossier.

    Parameters
    ----------
    dirpath :  String
        Un lien relatif ou absolu vers le dossier que l'on veut traiter.

    Returns
    -------
        Une liste contenant le chemin relatif de tout les fichiers dans ce dossier.
    """
    return [
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f))
    ]


def split_files(files, proportion_validation=0, proportion_test=0):
    """Sépare un jeu de données en jeux d'apprentissage, valisation et test.

    Parameters
    ----------
    files : List
        Liste de tout les fichiers à séparer.
    proportion_validation : int, optional
        Proportion du jeu à mettre en validation, par défaut 0.
    proportion_test : int, optional
        Proportion du jeu à mettre en test, par défaut 0.

    Returns
    -------
        Un tuple contenant 1, 2 ou 3 listes contenant les chemins vers les fichiers.
    """
    assert (
        proportion_test + proportion_validation < 1
    ), "La somme des proportions de validation et de test doit être inférieur à 1"
    coupe_1 = int(len(files) * round(1 - proportion_validation - proportion_test, 2))
    coupe_2 = int(len(files) * round(1 - proportion_test, 2))

    files_apprentissage = files[:coupe_1]
    files_validation = files[coupe_1:coupe_2]
    files_test = files[coupe_2:]

    return files_apprentissage, files_validation, files_test


def load_labels(filepath):
    labels = {}
    with open(filepath, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            labels[row["fichier"]] = int(row["label"])
    return labels
