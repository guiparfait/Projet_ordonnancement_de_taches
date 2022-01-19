# -*- coding=utf-8 -*-

"""
Skeleton for RO project: task scheduling with linear programming
Due for Monday the 6th of December 2021

Student 1:Ble Esolin Anderson
Student 2:GUI Parfait
"""

import sys
from _ast import Or
from datetime import time
from hashlib import scrypt
from pathlib import Path  # built-in usefull Path class
from argparse import ArgumentParser

from project_file_to_network import get_networks_data
from networkx import DiGraph, write_graphml_lxml, is_directed, generate_graphml
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpInteger,
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)
from project_file_to_network import (
    WEIGHT,
    ALPHA,
    OMEGA,
)


# ============================================================================ #
#                                    OTHERS                                    #
# ============================================================================ #
def parse_arguments():
    """Parse the input arguments and retrieve the choosen resolution method and
    the instance that must be solve."""
    argparser = ArgumentParser()
    argparser.add_argument(
        '--method', dest='method', required=False, default='PLC',
        help='Select the method to use',
    )

    argparser.add_argument(
        '--all-data', dest='all_data', action='store_true',
        help='Compute all the data',
    )

    arg = argparser.parse_args()

    if arg.method not in ['PLC', 'Bellman-Ford']:
        argparser.print_help()
        sys.exit(1)

    return (arg.method, arg.all_data)


# ============================================================================ #
#                               LP MODEL - PLC                                 #
# ============================================================================ #
def plc(graph: DiGraph, i: str, j: str) -> float:
    """The minimal duration to do a task j from a task i using PLC approach.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * i: the source node id (str)
    * j: the target node id (str)
    """

    # TODO: question 5
    prob, d_pos = model_heaviest_path_in_graph(graph, i, j)
    prob.solve(PULP_CBC_CMD(msg=False, logPath='./CBC_path.log'))

    lt_pos_nodes = sorted(
        [
            (pos.varValue, node) for node, pos in d_pos.items()
            if pos.varValue > 0
        ],
        key=lambda t: t[0],
    )
    # Each of the variables is printed with it's resolved optimum value
    # print('POSITION\tNODE')
    # for pos, node in lt_pos_nodes:
    #     print(f'{pos}\t\t{node}')
    return prob.objective.value()


def defining_path_variables(graph):
    """Return variables defining a path in a graph and big M constant."""
    # ------------------------------------------------------------------------ #
    #                              Vertex Position                             #
    # ------------------------------------------------------------------------ #
    d_pos = {}  # the dictionnary of LpVariable which means: the position of u
    for v in graph.nodes():
        # Position variable
        d_pos[v] = LpVariable(
            f'pos_{v}', lowBound=0, cat=LpInteger,
        )

    # ------------------------------------------------------------------------ #
    #                            Edges Binary Choice                           #
    # ------------------------------------------------------------------------ #
    d_x_e = {}  # a dictionnary of LpVariable which means: is e in path?
    for (u, v) in graph.edges():
        # Add the edges binaries
        d_x_e[(u, v)] = LpVariable(f'x_{u}_{v}', cat=LpBinary)
        if not is_directed(graph):
            # because in undirected graph we can go from both directions
            d_x_e[(v, u)] = LpVariable(f'x_{v}_{u}', cat=LpBinary)

    # Big M
    # -----
    big_m = graph.number_of_nodes()

    return d_pos, d_x_e, big_m


def define_path_weight(graph, prob, d_x_e):
    """Define path's weight."""
    prob += lpSum(
        x_e * graph.edges()[e][WEIGHT]
        for e, x_e in d_x_e.items()
    )
    return prob


def defining_a_path_from_source_to_target(graph, prob, d_pos, d_x_e, big_m,
                                          source, target):
    """Add constraints defining what a path in graph is."""
    # ------------------------------------------------------------------------ #
    #                        Elementary Path Definition                        #
    # ------------------------------------------------------------------------ #
    for u in graph.nodes():

        if u == source:
            # must begin by the source
            # ------------------------
            prob += d_pos[source] == 1, 'first_position'

            # source has exactly one successor in path
            # ----------------------------------------
            prob += lpSum(x_e for e, x_e in d_x_e.items() if e[0] == u) == 1

        elif u == target:
            # target has exactly one predecessor in path
            # ------------------------------------------
            prob += lpSum(x_e for e, x_e in d_x_e.items() if e[1] == u) == 1

        else:  # A possible intermediate vertex in path

            nb_preds = lpSum(
                x_e for e, x_e in d_x_e.items() if e[1] == u
            )

            # at most one predecessor
            # -----------------------
            prob += nb_preds <= 1, f'at_most_one_pred_for_{u}'

            # same number of predecessors as successors
            # -----------------------------------------
            prob += (
                nb_preds == lpSum(
                    x_e for e, x_e in d_x_e.items() if e[0] == u
                ),
                f'same_nb_preds_as_succs_for_{u}',
            )

            # vertex not in path => position = 0
            # ----------------------------------
            prob += (
                d_pos[u] <= nb_preds * big_m,
                f'{u}_not_kept_implies_pos_null',
            )

    # ------------------------------------------------------------------------ #
    #                 MTZ Constraints On Positions To Not Loop                 #
    # ------------------------------------------------------------------------ #
    for (u, v), x_e in d_x_e.items():
        # Be sure 'pos(u) -> pos_u' function is strictly increasing
        #   implies that loop are avoided
        prob += (
            d_pos[v] >= d_pos[u] + x_e - (1 - x_e) * big_m,
            f'pos_{v}_according_pos_{u}_1',
        )
        prob += (
            d_pos[v] <= d_pos[u] + x_e + (1 - x_e) * big_m,
            f'pos_{v}_according_pos_{u}_2',
        )
    return prob


def model_heaviest_path_in_graph(graph, source, target):
    """Find the heaviest path between a source and a target."""
    # ------------------------------------------------------------------------ #
    # Linear problem with minimization or maximization
    # ------------------------------------------------------------------------ #
    prob = LpProblem(name='heaviest_path', sense=LpMaximize)

    # ------------------------------------------------------------------------ #
    # The variables
    # ------------------------------------------------------------------------ #
    d_pos, d_x_e, big_m = defining_path_variables(graph)

    # ------------------------------------------------------------------------ #
    # The objective function
    # ------------------------------------------------------------------------ #
    prob = define_path_weight(graph, prob, d_x_e)

    # ------------------------------------------------------------------------ #
    # The constraints
    # ------------------------------------------------------------------------ #
    prob = defining_a_path_from_source_to_target(
        graph, prob, d_pos, d_x_e, big_m, source, target,
    )

    return prob, d_pos


def plc_compute_earliest_time(graph: DiGraph, i: str) -> float:
    """Return the earliest time to do the task i using the PLC method.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * i: node id (str)
    """
    # TODO: question 6
    return plc(graph, ALPHA, i)


def plc_compute_latest_time(graph: DiGraph, t_omega: float, i: str) -> float:
    """Return the latest time to do the task i using the PLC method.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * t_omega: the minimal duration to do the task omega (float)
    * i: node id (str)
    """
    # TODO: question 7
    return t_omega - plc(graph, i, OMEGA)


# ============================================================================ #
#                          LP MODEL - Bellman-Ford                             #
# ============================================================================ #

def bf_defining_path_variables(graph):
    """Return variables defining a path in a graph and big M constant."""
    # ------------------------------------------------------------------------ #
    #                              Vertex Position                             #
    # ------------------------------------------------------------------------ #
    d_pos = {}  # the dictionnary of LpVariable which means: the position of u
    for v in graph.nodes():
        # Position variable
        d_pos[v] = LpVariable(
            f'pos_{v}', lowBound=0, cat=LpInteger,
        )

    # ------------------------------------------------------------------------ #
    #                            Edges Binary Choice                           #
    # ------------------------------------------------------------------------ #
    d_x_e = {}  # a dictionnary of LpVariable which means: is e in path?
    for (u, v) in graph.edges():
        # Add the edges binaries
        d_x_e[(u, v)] = LpVariable(f'x_{u}_{v}', cat=LpBinary)
        if not is_directed(graph):
            # because in undirected graph we can go from both directions
            d_x_e[(v, u)] = LpVariable(f'x_{v}_{u}', cat=LpBinary)

    # Big M
    # -----
    big_m = graph.number_of_nodes()

    return d_pos, d_x_e, big_m


def bf_define_path_weight(graph, prob, d_x_e, d_pos):
    """Define path's weight."""
    prob += lpSum(
        pos
        for node, pos in d_pos.items()
    )
    return prob


def bf_defining_a_path_from_source_to_target(graph, prob, d_pos, d_x_e, big_m,
                                             source, target):
    """Add constraints defining what a path in graph is."""
    # ------------------------------------------------------------------------ #
    #                        Elementary Path Definition                        #
    # ------------------------------------------------------------------------ #
    for u in graph.nodes():

        if u == source:
            # must begin by the source
            # ------------------------
            prob += d_pos[source] == 1, 'first_position'

            # source has exactly one successor in path
            # ----------------------------------------
            prob += lpSum(x_e for e, x_e in d_x_e.items() if e[0] == u) == 1

        elif u == target:
            # target has exactly one predecessor in path
            # ------------------------------------------
            prob += lpSum(x_e for e, x_e in d_x_e.items() if e[1] == u) == 1

        else:  # A possible intermediate vertex in path

            nb_preds = lpSum(
                x_e for e, x_e in d_x_e.items() if e[1] == u
            )

            # at most one predecessor
            # -----------------------
            prob += nb_preds <= 1, f'at_most_one_pred_for_{u}'

            # same number of predecessors as successors
            # -----------------------------------------
            prob += (
                nb_preds == lpSum(
                    x_e for e, x_e in d_x_e.items() if e[0] == u
                ),
                f'same_nb_preds_as_succs_for_{u}',
            )

            # vertex not in path => position = 0
            # ----------------------------------
            prob += (
                d_pos[u] <= nb_preds * big_m,
                f'{u}_not_kept_implies_pos_null',
            )

    # ------------------------------------------------------------------------ #
    #                 MTZ Constraints On Positions To Not Loop                 #
    # ------------------------------------------------------------------------ #
    for (u, v), x_e in d_x_e.items():
        # graph.edges.date
        # Be sure 'pos(u) -> pos_u' function is strictly increasing
        #   implies that loop are avoided
        if u == source:
            prob += d_pos[source] == 0
        else:
            prob += (
                d_pos[u] <= d_pos[v] - x_e,
                f'pos_{u}_according_pos_{v}_1',
            )
        # prob += (
        #     d_pos[v] <= d_pos[u] + x_e + (1 - x_e) * big_m,
        #     f'pos_{v}_according_pos_{u}_2',
        # )

    # d_pos[ALPHA] = 0
    # prob += d_pos[ALPHA]
    return prob


def bf_model_heaviest_path_in_graph(graph, source, target):
    """Find the heaviest path between a source and a target."""
    # ------------------------------------------------------------------------ #
    # Linear problem with minimization or maximization
    # ------------------------------------------------------------------------ #
    prob = LpProblem(name='heaviest_time_alpha', sense=LpMaximize)

    # ------------------------------------------------------------------------ #
    # The variables
    # ------------------------------------------------------------------------ #
    d_pos, d_x_e, big_m = bf_defining_path_variables(graph)

    # ------------------------------------------------------------------------ #
    # The objective function
    # ------------------------------------------------------------------------ #
    prob = bf_define_path_weight(graph, prob, d_x_e, d_pos)

    # ------------------------------------------------------------------------ #
    # The constraints
    # ------------------------------------------------------------------------ #
    prob = bf_defining_a_path_from_source_to_target(
        graph, prob, d_pos, d_x_e, big_m, source, target,
    )

    return prob, d_pos


def bf(graph: DiGraph, source: str, target: str) -> float:
    prob, d_pos = bf_model_heaviest_path_in_graph(graph, source, target)
    prob.solve(PULP_CBC_CMD(msg=False, logPath='./CBC_path.log'))

    lt_pos_nodes = sorted(
        [
            (pos.varValue, node) for node, pos in d_pos.items()
            if pos.varValue > 0
        ],
        key=lambda t: t[0],
    )
    # Each of the variables is printed with it's resolved optimum value
    print('POSITION\tNODE')
    for pos, node in lt_pos_nodes:
        print(f'{pos}\t\t{node}')
    return prob.objective.value()


def bf_compute_earliest_time(graph: DiGraph, i: str) -> float:
    """Return the earliest time to do the task i using the Bellman-Ford
    method.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * i: node id (str)
    """
    # TODO: question 9
    return bf(ALPHA, i)


def bf_compute_latest_time(graph: DiGraph, t_omega: float, i: str) -> float:
    """Return the latest time to do the task i using the Bellman-Ford
    method.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * t_omega: the minimal duration to do the task omega (float)
    * i: node id (str)
    """
    # TODO: question 11


# ============================================================================ #
#                                Task Scheduling                               #
# ============================================================================ #
def solve_task_scheduling(method: str, graph: DiGraph) -> dict:
    """Compute the couple (t_i, T_i), earliest and latest time, for each task i
    and return the scheduling as a dictionnary.
    Be careful to use the selected method to compute the scheduling.
    ARGUMENTS:
    ----------
    * method: name of the method that must be used for the resolution of the
        tasks scheduling problem: either PLC or Bellman-Ford (str)
    * graph: a potential-task graph (networkx.DiGraph)
    """
    # TODO: question 12
    scheduling = {}
    if method.lower() == 'plc':
        for i in graph.nodes():
            if i != ALPHA and i != OMEGA:
                ti = plc_compute_earliest_time(graph, i)
                t_omega = plc(graph, ALPHA, OMEGA)
                Ti = plc_compute_latest_time(graph, t_omega, i)
                scheduling[i] = (ti, Ti)
    elif method.lower() == 'bf':
        for i in graph.nodes():
            continue
            # ti = bf_compute_earliest_time(graph, i)
            # Ti = bf_compute_latest_time(graph, i)
            # scheduling[i] = (ti, Ti)
    return scheduling
    # print(bf(graph, "4", "8"))


# ============================================================================ #
#                              Extended Analysis                               #
# ============================================================================ #
def compute_slack(graph: DiGraph, scheduling: dict, i: str) -> float:
    """Return the slack value of the task i according to the scheduling.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * scheduling: a scheduling (dict such that foreach node i {i: (t_i, T_i)})
    * i: node id (str)
    """
    # TODO: question 14
    ti, Ti = scheduling[i]
    return Ti - ti


def compute_free_slack(graph: DiGraph, scheduling: dict, i: str) -> float:
    """Return the free slack value of the task i according to the scheduling.
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * scheduling: a scheduling (dict such that foreach node i {i: (t_i, T_i)})
    * i: node id (str)
    """
    # TODO: question 14
    liste = []
    for (u, v, du) in graph.edges.data(WEIGHT):
        if u != ALPHA and v != OMEGA:
            ti, _ = scheduling[i]
            tv, _ = scheduling[v]

            if i == u:
                liste.append(tv - ti - du)
    return min(liste) if len(liste) > 0 else 0


def solve_task_scheduling_extended(graph: DiGraph, scheduling: dict) -> dict:
    """Compute the couple (t_i, T_i, mt_i, ml_i), earliest time, latest time,
    slack and free slack, for each task i and return it as a dictionary
    ARGUMENTS:
    ----------
    * graph: a potential-task graph (networkx.DiGraph)
    * scheduling: a scheduling (dict such that foreach node i {i: (t_i, T_i)})
    """
    # TODO: question 15
    sche = {}
    for i, (ti, Ti) in scheduling.items():
        mTi = compute_slack(graph, scheduling, i)
        mli = compute_free_slack(graph, scheduling, i)
        sche[i] = (ti, Ti, mTi, mli)
    return sche


# ============================================================================ #
#                                 Visualisation                                #
# ============================================================================ #
def visualize_task_scheduling_results(file_path: str, graph: DiGraph,
                                      extended_scheduling: dict) -> None:
    """Create a GraphML file at file_path containing all the necessary data to 
    correctly visualise the scheduling.
    ARGUMENTS:
    ----------
    * file_path: path to the GraphML file to create
    * graph: a potential-task graph (networkx.DiGraph)
    * extended_scheduling: a scheduling 
        (dict such that foreach node i {i: (t_i, T_i)})
    """
    # TODO: question 17
    file = open(file_path, "w")
    file.write("Tache i \t ti  \t Ti \t \n")
    for i, (ti, Ti) in extended_scheduling.items():
        file.write(f"{i}\t\t  {ti}\t\t  {Ti} ")
        file.write("\n")
    file.close()


# ============================================================================ #
#                                   UTILITIES                                  #
# ============================================================================ #
def print_log_output(prob: LpProblem) -> None:
    """Print the log output and problem solutions.
    ARGUMENTS:
    ----------
    * prob: an solved LP model (pulp.LpProblem)
    """
    print()
    print('-' * 40)
    print('Stats')
    print('-' * 40)
    print()
    print(f'Number variables: {prob.numVariables()}')
    print(f'Number constraints: {prob.numConstraints()}')
    print()
    print('Time:')
    print(f'- (real) {prob.solutionTime}')
    print(f'- (CPU) {prob.solutionCpuTime}')
    print()

    print(f'Solve status: {LpStatus[prob.status]}')
    print(f'Objective value: {prob.objective.value()}')

    print()
    print('-' * 40)
    print("Variables' values")
    print('-' * 40)
    print()
    # TODO: you can print variables value here.


if __name__ == '__main__':

    # QUESTION 2
    write_graphml_lxml(next(get_networks_data()), Path('./data/task_scheduling.graphml'))

    # # Read the arguments to select the resolution method
    selected_method, use_all_data = parse_arguments()

    for graph in get_networks_data(use_all_data):
        # TODO: complete this according to your needs.

        scheduling = solve_task_scheduling(selected_method, graph)

        # print(plc(graph,"4","8"))
        # print(plc_compute_earliest_time(graph,"4"))
        # ti = plc(graph,ALPHA,OMEGA)
        # print(plc_compute_latest_time(graph,ti,"4"))
        scheduling_extended = solve_task_scheduling_extended(graph, scheduling)
        generate_graphml(visualize_task_scheduling_results("./data/task_new_.data", graph, scheduling))
        # [print(node, " => ", tt) for node, tt in scheduling.items()]
        #
        print("-----------------------------------------------")
        print("Tâche i \t ti \t Ti \t mTi \t mli \t \t")
        print("-----------------------------------------------")
        for i, (ti, Ti, mTi, mli) in scheduling_extended.items():
            print(f"{i} \t {ti} \t {Ti} \t {mTi} \t {mli} \t")
        print("-----------------------------------------------")
