import pandas as pd
from dataclasses import dataclass
from numpy.random import default_rng
from DaO_simulation import dao
from SpatialAnalytics import boss
import numpy as np
import secrets


@dataclass
class SEM:
    nvars: int = 10
    nrows: int = 50
    avg_deg: int = 2
    seed: int = secrets.randbits(63)
    directed: bool = True
    debug: bool = False
    pd: int = 64
    #   G: np.array = None

    def __post_init__(self):
        self.names = [f"X_{x}" for x in range(self.nvars)]
        self.rng = default_rng(self.seed)
        self.make_graph()
        self.make_data()

        # provides a dataframe version of the beta matrix created by DaO
        # ((adds column and index labels))
        self.Beta_df = pd.DataFrame(
            self.B, columns=self.names, index=self.names)

        self.results = boss.Boss(
            self.data, penalty_discount=self.pd, seed=self.seed)

    #       self.discovered_directed_edges, self.discovered_undirected_edges = results

    def make_graph(self):
        """
        Create a DAG self.G,
        and from it create the correlation matrix self.R,
        the beta matrix self.B,
        and the error vector self.o
        """
        g = dao.er_dag(self.nvars, ad=self.avg_deg, rng=self.rng)
        g = dao.sf_out(g, rng=self.rng)
        self.G = dao.randomize_graph(g, rng=self.rng)
        self.R, self.B, self.o = dao.corr(self.G, rng=self.rng)

    def make_data(self):
        """Generate self.nrows of data from the graph self.G"""
        X = dao.simulate(self.B, self.o, self.nrows, rng=self.rng)
        X = dao.standardize(X)  # no rng parameter
        self.data = pd.DataFrame(X, columns=self.names)

    def get_edges(self):
        el = []
        df = pd.DataFrame(self.G, index=self.names, columns=self.names)
        for v in df.columns:
            for u in df[df[v] != 0].index:
                el.append((u, v))
        return el

    def true_edge_df(self):
        true_edges = set(self.get_edges())
        directed_edges = self.results[0]
        undirected_edges = self.results[1]
        edf = pd.DataFrame(index=range(len(true_edges)))
        edf["seed"] = self.seed
        edf["nrows"] = self.nrows
        edf["nvars"] = self.nvars
        edf["avg_deg"] = self.avg_deg
        edf["u"] = [u for u, v in true_edges]
        edf["v"] = [v for u, v in true_edges]
        edf["Beta"] = [self.Beta_df.at[u, v] for u, v in true_edges]
        edf["TrueEdge"] = [1 for e in true_edges]
        discovered = []
        oriented = []
        correctly_oriented = []
        for u, v in true_edges:
            if (u, v) in directed_edges:
                discovered.append(1)
                oriented.append(1)
                correctly_oriented.append(1)
            elif (v, u) in directed_edges:
                discovered.append(1)
                oriented.append(1)
                correctly_oriented.append(0)
            elif (u, v) in undirected_edges or (v, u) in undirected_edges:
                discovered.append(1)
                oriented.append(0)
                correctly_oriented.append(0)
            else:
                discovered.append(0)
                oriented.append(0)
                correctly_oriented.append(0)
        edf["Discovered"] = discovered
        edf["Oriented"] = oriented
        edf["Correctly Oriented"] = correctly_oriented
        edf["outdeg"] = 0
        edf["indeg"] = 0
        edf["penalty_discount"] = self.pd
        for u in edf.u.unique():
            edf.loc[edf.u == u, "outdeg"] = len(edf[edf.u == u])
        for v in edf.v.unique():
            edf.loc[edf.v == v, "indeg"] = len(edf[edf.v == v])

        return edf

    def get_betas(l, B):
        return [B.at[u, v] for u, v in l]

    def evaluate_edges(self, discovered_edges, directed=True):
        true_edges = self.get_edges()
        if directed:
            tp = [x for x in discovered_edges if x in true_edges]
            fn = [x for x in true_edges if x not in discovered_edges]
            fp = []
        else:
            tp = []
            fn = []
            for u, v in true_edges:
                if (u, v) in discovered_edges:
                    tp.append((u, v))
                elif (v, u) in discovered_edges:
                    tp.append((u, v))
                else:
                    fp.append((u, v))

        #   tp_beta = [B.at[u,v] for u,v in tp]
        #   fn_beta = [B.at[u,v] for u,v in fn]

        return tp, fn

    def simulate(self, debug=False):
        dde, due = boss.Boss(self.data, penalty_discount=64)
        if debug:
            print("Discovered edges:\n", dde, "\n", due)
        return

        if self.directed:
            tp_edges, fn_edges = self.evaluate_edges(self.Beta_df, dde)
        else:
            tp_edges, fn_edges = self.evaluate_edges(Beta_df, dde + due)
        if debug:
            print("\nTrue Positives", tp_edges, "\nFalse Negatives", fn_edges)
        #   return evaluate(b, de)
        return get_betas(tp_edges, Beta_df), get_betas(fn_edges, Beta_df)
