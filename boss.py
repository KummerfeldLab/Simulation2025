import jpype
import jpype.imports
#import semopy
from pathlib import Path
jpype.startJVM("-Xmx16g", classpath=Path.home() / "tetrad-gui-7.6.6-launch.jar")

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts

#from pytetrad.tools import TetradSearch as ts

def df_to_boxdataset(df):
    """ By Bryan Andrews
        Converts a pandas DataFrame to a Java BoxDataSet
    """
    cols = df.columns
    values = df.values
    n, p = df.shape

    variables = util.ArrayList()
    for col in cols: 
        variables.add(td.ContinuousVariable(str(col)))

    databox = td.DoubleDataBox(n, p)
    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    return td.BoxDataSet(databox, variables)


def Boss(df, penalty_discount=64, shutdown=False, seed=None):
    """ Searches a dataframe for a graph, returned as a pair of edgelists.
        Each edglist consists of (u,v) pairs.
        The first edgelist is directed, indicating  u --> v,
        The second edgelist is undirected, indicating u --- v """
    # SEARCH PARAMETERS
    boss_bes = False
    boss_starts = 1
    boss_threads = 8

    data = df_to_boxdataset(df)

    # print("Setting up SEM-BIC Score")
    score = ts.score.SemBicScore(data, True)
    score.setPenaltyDiscount(penalty_discount)
    score.setStructurePrior(0)

    # print("Setting up BOSS")
    boss = ts.Boss(score)
    boss.setUseBes(boss_bes)
    boss.setNumStarts(boss_starts)
    boss.setNumThreads(boss_threads)
    boss.setUseDataOrder(False)
    boss.setResetAfterBM(True)  # SET TO TRUE IF HAVING MEMORY ISSUES
    boss.setResetAfterRS(False)
    boss.setVerbose(False)

    # print("Running BOSS")
    boss = ts.PermutationSearch(boss)
    boss.setSeed(seed)
    graph = boss.search()
    el = edgelistFromTetrad(graph)
    if shutdown:
        jpype.shutdownJVM()
    return el

# end from Bryan Andrews


def edgelistFromTetrad(graph, debug=False):
    if debug:
        print("\nProcessing Edgelist\n")
#   graph = tg.GraphTransforms.dagFromCpdag(graph)
    el = []
    ul = []
    if debug:
        print('Interating over edge')
    for e in graph.getEdges():
        if debug:
            print(e)
        u, d, v = str(e).split(' ')
        if d == '---':
            ul.append((u, v))
#           print('Undirected edge', e)
        else:
            el.append((u, v))
    return el, ul


def get_causal_matrix(df):
    search = ts.TetradSearch(df)
    search.set_verbose(False)
    search.use_sem_bic()
    search.use_fisher_z(alpha=0.05)
    search.run_boss(num_starts=1, use_bes=True, time_lag=0,
                    use_data_order=True)
    graph = search.get_graph_to_matrix()
    return graph


def edgelist(edgematrix, source='MATLAB'):
    edgelist = set()
    if source == 'MATLAB':
        outedge, inedge = (1, 2)
    else:
        outedge, inedge = (2, 3)
    row_sum = edgematrix.sum(axis=1)
    non_empty_rows = row_sum[row_sum > 0].index
    for index in non_empty_rows:
        row = edgematrix.loc[index]
        for target in row[row == outedge].index:
            edgelist.add((index, target))
        for source in row[row == inedge].index:
            edgelist.add((source, index))
    return list(edgelist)


def predecessors(edgelist):
    preds = {}
    for source, target in edgelist:
        if target not in preds:
            preds[target] = [source]
        else:
            preds[target].append(source)
    return preds


def successors(edgematrix):
    gsum = edgematrix.sum(axis=1)
    succs = {col: [] for col in edgematrix.columns}
    for idx in gsum[gsum > 0].index:
        row = edgematrix.loc[idx]
        onecols = row[row == 2].index
        for col in onecols:
            succs[edgematrix.columns[idx]].append(col)
    return succs


def modelstrings(preds):
    modelstringlist = []
    for k, vl in preds.items():
        if len(vl) == 0:
            continue
        modelstringlist.append(k + ' ~ ' + ' + '.join(vl))
    modelstring = '\n '.join(modelstringlist)
    return modelstring


def SEM(df):
    M = get_causal_matrix(df)
    mod = semopy.Model(modelstrings(M))
    mod.fit(df)
    return M, mod
