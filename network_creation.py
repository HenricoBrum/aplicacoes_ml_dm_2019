import itertools
import igraph
from igraph import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Code for Visibility Network
class VNetwork(object):

    def __init__(self, network=None, time_serie=None, output='network.NET'):
        self.time_serie = time_serie
        if network is None:
            self.network = None
            self.N = len(self.time_serie)
        else:
            self.network = igraph.read(network, format="pajek")
            self.N = self.network.vcount()
        self.output = output

    def exists_node(self, y_a, y_b, t_a, t_b, tc_intermediates, values):
        for tc in tc_intermediates:
            yc = values[tc]
            operation = y_b + (y_a - y_b) * ((t_b - tc) / float(t_b - t_a))
            if yc > operation:
                return False
        return True

    def create_network(self):
        print 'Creating network.....'
        eje_x = [x for x in range(self.N)]
        all_edges = itertools.combinations([x for x in range(self.N)], 2)
        edges = []
        for edge in all_edges:
            t_a = edge[0]
            t_b = edge[1]
            y_a = self.time_serie[t_a]
            y_b = self.time_serie[t_b]
            if t_b - t_a == 1:
                edges.append((t_a, t_b))
            else:
                intermediates = eje_x[t_a + 1:t_b]
                if self.exists_node(y_a, y_b, t_a, t_b, intermediates, self.time_serie):
                    edges.append((t_a, t_b))
        network = Graph()
        network.add_vertices(self.N)
        network.add_edges(edges)
        print 'Network created.....'
        network.write_pajek(self.output)

    def get_network_measurements(self):
        average_degree = np.mean(self.network.degree())
        average_clustering_coefficient = self.network.transitivity_avglocal_undirected()
        average_path_length = self.network.average_path_length()
        communities = self.network.community_multilevel()
        number_communities = len(communities)
        modularity = self.network.modularity(communities)
        measurements = []
        measurements.append(average_degree)
        measurements.append(average_clustering_coefficient)
        measurements.append(average_path_length)
        measurements.append(number_communities)
        measurements.append(modularity)
        return measurements

    def get_network_measurements_normalized(self):
        measurements = self.get_network_measurements()
        max_value = max(measurements)
        min_value = min(measurements)
        normalized = []
        for measure in measurements:
            value = (measure - min_value) / float(max_value - min_value)
            normalized.append(value)
        return normalized


# Code for k-NN Network
class KNNNetwork(object):

    def __init__(self, k, output):
        self.k = k
        self.output = output

    def mydist(self,x, y):
        distance, path = fastdtw(x, y, dist=euclidean)
        return distance

    def create_network(self):
        data_path = 'all_stocks_5yr.csv'
        data = pd.read_csv(data_path)
        Name = data['Name']
        companies = list(set(Name))
        time_series = []
        valid_companies = []
        for index, company in enumerate(companies):
            all_time_series = data.loc[data['Name'] == company]
            ts_open = np.array(all_time_series['open'])
            ts_open = ts_open[~np.isnan(ts_open)]
            size = ts_open.shape[0]
            if size > 1100:
                valid_companies.append(company)
                ts_open.resize(1259)
                time_series.append(ts_open)
        time_series = np.array(time_series)
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', metric=self.mydist).fit(time_series)
        knn_graph = nbrs.kneighbors_graph(time_series).toarray()
        np.fill_diagonal(knn_graph, 0)
        g = igraph.Graph.Adjacency(knn_graph.tolist(), mode="undirected")
        print 'Network created.....'
        g.write_pajek(self.output)
