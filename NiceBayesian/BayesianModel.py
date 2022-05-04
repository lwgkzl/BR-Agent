#!/usr/bin/env python3

import itertools
from collections import defaultdict
import logging
from operator import mul
from functools import reduce
import torch
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pgmpy.base import DAG
from pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.models.MarkovModel import MarkovModel
from NiceBayesian.CPD import TabularCPD
import torch
from torch import nn

#
class MYBayesianModel(DAG, nn.Module):
    """
    Base class for Bayesian Models.
    """

    def __init__(self, ebunch=None, cpds=None):
        DAG.__init__(self)
        nn.Module.__init__(self)
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_edge(self, u, v, **kwargs):
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
            raise ValueError(
                "Loops are not allowed. Adding the edge from (%s->%s) forms a loop."
                % (u, v)
            )
        else:
            super(MYBayesianModel, self).add_edge(u, v, **kwargs)

    def add_cpds(self, cpds):
        for cpd in cpds:
            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(f"Replacing existing CPD for {cpd.variable}")
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None):
        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            else:
                for cpd in self.cpds:
                    if cpd.variable == node:
                        return cpd
        else:
            return self.cpds

    def get_cardinality(self, node=None):
        if node:
            return self.get_cpds(node).cardinality[0]
        else:
            cardinalities = defaultdict(int)
            for cpd in self.cpds:
                cardinalities[cpd.variable] = cpd.cardinality[0]
            return cardinalities

    def check_model(self):
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        f"CPD associated with {node} doesn't have proper parents associated with it."
                    )
        return True

    def predict(self, data, n_jobs=-1):
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        data_unique = data.drop_duplicates()
        missing_variables = set(self.nodes()) - set(data_unique.columns)

        # Send state_names dict from one of the estimated CPDs to the inference class.
        model_inference = VariableElimination(self)
        pred_values = Parallel(n_jobs=n_jobs)(
            delayed(model_inference.map_query)(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            for index, data_point in tqdm(
                data_unique.iterrows(), total=data_unique.shape[0]
            )
        )

        df_results = pd.DataFrame(pred_values, index=data_unique.index)
        data_with_results = pd.concat([data_unique, df_results], axis=1)
        return data.merge(data_with_results, how="left").loc[:, missing_variables]

    def predict_probability(self, data):
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for _, data_point in data.iterrows():
            full_distribution = model_inference.query(
                variables=missing_variables,
                evidence=data_point.to_dict(),
                show_progress=False,
            )
            states_dict = {}
            for var in missing_variables:
                states_dict[var] = full_distribution.marginalize(
                    missing_variables - {var}, inplace=False
                )
            for k, v in states_dict.items():
                for l in range(len(v.values)):
                    state = self.get_cpds(k).state_names[k][l]
                    pred_values[k + "_" + str(state)].append(v.values[l])
        return pd.DataFrame(pred_values, index=data.index)
