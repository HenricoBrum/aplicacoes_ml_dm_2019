import network
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import igraph
from igraph import *

def read_names_and_sectors():
    path_names = 'company_names.txt'
    path_sectors = 'company_sectors.txt'

    doc1 = open(path_names, 'r')
    doc2 = open(path_sectors, 'r')

    names = dict()
    sectors = dict()
    sector_labels = dict()
    index = 0

    for line1, line2 in zip(doc1, doc2):
        line1 = line1.rstrip('\n')
        line2 = line2.rstrip('\n')
        names[line1] = line2
        if line2 in sectors:
            sectors[line2].append(line1)
        else:
            sectors[line2] = [line1]
            sector_labels[line2] = index
            index+=1

    return [names, sectors, sector_labels]

def load_valid_companies():
    path = 'valid_companies.txt'
    document = open(path, 'r')
    companies = []
    for line in document:
        line = line.rstrip('\n')
        companies.append(line)
    return companies


class NetworkAnalysis(object):

    def __init__(self):
        self.data_path = 'all_stocks_5yr.csv'
        self.time_series_data = pd.read_csv(self.data_path)
        self.time_series_path = 'all_stocks_5yr.csv'
        self.open_network_1st = 'networks/1_open/'
        self.close_network_1st = 'networks/1_close/'
        self.high_network_1st = 'networks/1_high/'
        self.low_network_1st = 'networks/1_low/'

    def network_correlation(self):
        companies_data = read_names_and_sectors()
        company_names = companies_data[0]
        sector_labels = companies_data[2]

        path_networks = os.listdir(self.open_network_1st)

        results_first = [[] for _ in range(5)]
        results_second = [[] for _ in range(5)]

        for index, path in enumerate(path_networks):
            path_open = self.open_network_1st + path
            path_close = self.close_network_1st + path

            path_high = self.high_network_1st + path
            path_low = self.low_network_1st + path

            comp_name = path[:path.rfind('.NET')]
            sector = company_names[comp_name]
            label = sector_labels[sector]
            print index + 1, path_high, path_low, sector, label

            #obj = network.CNetwork(network=path_open)
            obj = network.CNetwork(network=path_high)
            measurements_1 = obj.get_network_measurements()

            #obj = network.CNetwork(network=path_close)
            obj = network.CNetwork(network=path_low)
            measurements_2 = obj.get_network_measurements()

            for i, (o, c) in enumerate(zip(measurements_1, measurements_2)) :
                results_first[i].append(o)
                results_second[i].append(c)

        measurements = ['avg degree' , 'avg cc', 'avg path', 'nro comm' , 'modularity']
        print ''
        for measure, r_1, r_2 in zip(measurements, results_first, results_second):
            print measure
            print 'Pearson correlation:' , pearsonr(r_1, r_2)[0]
            print 'Mean high:' , np.mean(r_1)
            print 'Mean low:', np.mean(r_2)
            print 'Std high:' , np.std(r_1)
            print 'Std low:', np.std(r_2)
            print ''

    def distance_analysis(self):
        data = self.time_series_data
        Name = data['Name']
        companies = list(set(Name))
        distances_o_c = []
        distances_h_l = []
        for index, company in enumerate(companies):
            print index + 1, company
            all_time_series = data.loc[data['Name'] == company]
            ts_open = np.array(all_time_series['open'])
            ts_open = ts_open[~np.isnan(ts_open)]
            ts_close = np.array(all_time_series['close'])
            ts_close = ts_close[~np.isnan(ts_close)]
            ts_high = np.array(all_time_series['high'])
            ts_high = ts_high[~np.isnan(ts_high)]
            ts_low = np.array(all_time_series['low'])
            ts_low = ts_low[~np.isnan(ts_low)]

            distance_o_c, path = fastdtw(ts_open, ts_close, dist=euclidean)
            distance_h_l, path = fastdtw(ts_high, ts_low, dist=euclidean)
            distances_o_c.append(distance_o_c)
            distances_h_l.append(distance_h_l)

        print np.max(distances_o_c), np.min(distances_o_c), np.mean(distances_o_c), np.std(distances_o_c)
        print np.max(distances_h_l), np.min(distances_h_l), np.mean(distances_h_l), np.std(distances_h_l)

        eje_x = [x for x in range(len(companies))]
        legends = ['Open vs Close', 'High vs Low']

        plt.scatter(eje_x, distances_o_c, c='blue', s=20)
        plt.scatter(eje_x, distances_h_l, c='red', s=20)
        plt.legend(legends, loc='upper right')
        plt.xlabel('Companies from SP-500')
        plt.ylabel('Distances')
        plt.title('Comparing the distances between Open vs Close and High vs Low values')
        plt.show()

    def knn_network_analysis(self):
        path = 'networks/2-open/10NN.NET'
        rede = igraph.read(path, format="pajek")
        valid_companies = load_valid_companies()
        companies_data = read_names_and_sectors()
        company_names = companies_data[0]
        sector_labels = companies_data[2]
        sector_colors = dict()
        sector_colors['Communication Services'] = 'blue'
        sector_colors['Industrials'] = 'red'
        sector_colors['Consumer Discretionary'] = 'green'
        sector_colors['Utilities'] = 'rgb(64%, 0%, 0%)'
        sector_colors['Consumer Staples'] = 'rgb(84%, 242%, 0%)'
        sector_colors['Health Care'] = 'rgb(95%, 2%, 154%)'
        sector_colors['Materials'] = 'rgb(166%, 123%, 3%)'
        sector_colors['Financials'] =  'rgb(23%, 190%, 207%)'
        sector_colors['Real Estate'] = 'rgb(255%, 127%, 80%)'
        sector_colors['Energy'] = 'rgb(20%, 128%, 128%)'
        sector_colors['Information Technology'] = 'rgb(218%, 165%, 32%)'

        print company_names
        print sector_labels
        print

        comunidades = rede.community_multilevel()
        #comunidades = rede.community_label_propagation()
        #comunidades = rede.community_infomap()

        clustered_companies = []
        for community in comunidades:
            data = []
            companies = []
            for company in community:
                data.append(company_names[valid_companies[company]])
                print valid_companies[company],
                companies.append(valid_companies[company])
            clustered_companies.append(companies)

        plot(comunidades, mark_groups=True)
        print 'nro de comunidades' , len(comunidades)
        colors = []
        for vertex , company in zip(rede.vs, valid_companies):
            colors.append(sector_colors[company_names[company]])
        rede.vs["color"] = colors



if __name__ == '__main__':

    obj = NetworkAnalysis()
    obj.knn_network_analysis()
    #obj.network_correlation()
