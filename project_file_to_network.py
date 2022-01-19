# -*- coding=utf-8 -*-

"""Project: Module to extract networks from files."""

from argparse import ArgumentParser
from pathlib import Path
from networkx import DiGraph, write_graphml_lxml

from typing import Tuple, Dict, List

# ============================================================================ #
#                                   CONSTANTS                                  #
# ============================================================================ #
# tache de debut
ALPHA = 'ALPHA'
# tache de fin
OMEGA = 'OMEGA'
# durée des tâches
WEIGHT = 'weight'


# ============================================================================ #
#                                   FUNCTION                                   #
# ============================================================================ #
def extract_network(file_path: str) -> DiGraph:
    """Return a NetworkX DiGraph modelling the data.
    ARGUMENTS:
    ----------
    * file_path: path to the file that must be parsed (str)
    """
    # Tous les arcs du graphe
    edgesGraph = []  # type: List[Tuple[str, str, Dict[str, int]]]

    listAnt, dictNodeList = get_nodes_list(file_path)

    # tache de debut
    # ALPHA = 'ALPHA'
    # tache de fin
    # OMEGA = 'OMEGA'

    for node, values in dictNodeList.items():
        duree, ant = values

        if len(ant) == 0:
            #  Lier les noeuds avec ALPHA si pas de predecesseur
            i = ALPHA
            j = node
            di = 0
            edgesGraph.append(
                (i, j, {WEIGHT: di})
            )
        else:
            # Ajouter les noeuds entre les taches
            for previous_node in ant:
                i = previous_node
                j = node
                di, _ = dictNodeList[previous_node]

                edgesGraph.append(
                    (i, j, {WEIGHT: int(di)})
                )

        #  Lier les noeuds sans successeurs avec OMEGA
        if node not in listAnt:
            i = node
            j = OMEGA
            di, _ = dictNodeList[node]

            edgesGraph.append(
                (i, j, {WEIGHT: int(di)})
            )

    # Creation du Graph NetworkX
    graph = DiGraph()
    graph.add_edges_from(
        edgesGraph
    )
    return graph


def get_nodes_list(filepath: str) -> Tuple[List[str], Dict[str, Tuple[int, List[str]]]]:
    listAnt = []
    dictNodeList = {}

    with open(filepath) as file:
        for line in file:
            node, _, duree, ant = line.replace('\n', '').split('\t')
            ant = ant.split(',')

            # si la chaine est vide, le tableau des predecesseurs est vide
            if ant[0] == '':
                ant = []

            listAnt.extend(ant)
            dictNodeList[node] = (duree, ant)

    return listAnt, dictNodeList


# ---------------------------------------------------------------------------- #
#                                Argument Parser                               #
# ---------------------------------------------------------------------------- #
def get_networks_data(all_data: bool = False) -> DiGraph:
    """Iterate over directed graph built from data. Iterate over only one file
    if all_data is False, else iterate over all the files.
    ARGUMENTS:
    ----------
    * all_data: if True, iterate over all the instances, else only the first one
        DEFAULT: False
    """

    if all_data:
        for k in range(1, 5):
            file = Path(f'data/task_scheduling_{k}.data')
            print()
            print(f'== FILE: {file.name} ==')
            print()
            yield extract_network(file)
    else:
        file = Path('data/task_scheduling.data')
        print()
        print(f'== FILE: {file.name} ==')
        print()
        yield extract_network(file)


# ============================================================================ #
#                                     MAIN                                     #
# ============================================================================ #
if __name__ == '__main__':
    # TODO: test here your extracting function(s)
    graph = extract_network('./data/task_scheduling.data')
    for e, w in graph.edges().items():
        print(e, w)
    write_graphml_lxml(next(get_networks_data()), './task_scheduling.data')
    pass
