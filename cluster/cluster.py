import pandas as pd
import numpy as np
from os import path, mkdir
from colorama import init, Fore
from copy import deepcopy
from matplotlib import pyplot as plt
from typing import Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

# system settings
import sys
import argparse


sys.tracebacklimit = 0
init(convert=True, autoreset=True)


class Cluster(object):

    def __init__(self,
                dataset_path: str, 
                num_inits: int = 1,
                new_csv: bool = False,
                new_dir: str = '',
                verbose: bool = False) -> None:
        # checks
        if not path.exists(dataset_path):
            raise FileNotFoundError(Fore.RED + 'The path provided is incorrect')
        if not path.isfile(dataset_path):
            raise FileNotFoundError(Fore.RED + 'The path provided is not a file')
        
        # extract data based on path
        file_string = dataset_path.split('/')
        self.__filename = file_string[-1]
        self.__data_directory = '.' if len(file_string) < 2 else file_string[-2]
        self.__extension = self.__filename.split('.')[-1]
        self.__filename = self.__filename.split('.')[0]

        if self.__extension != 'csv':
            raise ValueError(Fore.RED + 'File is not a csv')
        
        if num_inits <= 0:
            raise ValueError(Fore.RED + 'There must be at least one initialization')
        
        # meta data for new file
        self.__create_new = new_csv
        self.__new_data_dir = new_dir if self.__create_new else '%s' % self.__data_directory

        # other meta data
        self.__verbose = verbose
        self.__df = None
        self.__bins = 45
        self.__distance_estimate = 0
        self.__mem_bins = 100
        self.__num_inits = num_inits
        self.__exclusion = 0
        self.__top_mem = 0
        self.__gmm = GaussianMixture(n_components=2, n_init=self.__num_inits)
        self.__min_samples = 0
        self.__epsilon = 0
        self.__dbscan = None 


        self.ra_lab = r'Right Ascension (deg) [$\alpha$]'
        self.dec_lab = r'Declination (deg) [$\delta$]'
        self.pmra_lab = r'Proper Motion in Right Ascension (mas/yr) [$\mu_{\alpha*}$]'
        self.pmdec_lab = r'Proper Motion in Declination (mas/yr) [$\mu_\delta$]'
        self.axes = dict(gaia=(r'$G-R_P$', r'$G$'), sdss=(r'$G-R$', r'$G$'), simbad=(r'$B-V$', r'$B$'))
        self.cmd = dict(gaia=('g_rp', 'g'), sdss=('g','r','g'), simbad=('B','V','B'))
        self.plots = ['hist', 'hist2d', 'plot', 'scatter', 'boxplot']
        self.mem_cols = ['pmra', 'pmdec', 'parallax']
        self.__slopes = [0.719, 1.392]
        self.__intercepts = [9.37, -84.5]

        keys = self.axes.keys()
        match = False
        for key in keys:
            if key in self.__filename:
                match = True
                break
        
        if not match:
            raise ValueError(Fore.RED + 'Data is not recognized as one of supported types: ' + str([x for x in keys]))
        

    def load_data(self,
                inplace: bool = True) -> Union[pd.DataFrame, None]:
        # load data into a Dataframe
        self.__df = pd.read_csv('%s/%s.%s' % (self.__data_directory, self.__filename, self.__extension))
        if not inplace:
            return deepcopy(self.__df)
        return None
        
    
    def __str__(self) -> str:
        # print class data
        to_ret = '========== CLUSTER INSTANCE META DATA ==========\n\n'

        to_ret += Fore.WHITE + 'Data Source File Name: \t\t' + Fore.CYAN + '%s\n' % self.__filename
        to_ret += Fore.WHITE + 'Data Source Directory: \t\t' + Fore.CYAN + '%s' % self.__data_directory
        if self.__data_directory == '.':
            to_ret += ' (root)'
        to_ret += '\n'
        
        to_ret += Fore.WHITE + 'Data Source File Extension: \t' + Fore.CYAN + '%s\n' % self.__extension
        to_ret += Fore.WHITE + 'Create Data Output File: \t'
        if self.__create_new:
            to_ret += Fore.GREEN + 'True'
        else:
            to_ret += Fore.RED + 'False'
        to_ret += '\n'
        
        to_ret += Fore.WHITE + 'Data Output Directory: \t\t' + Fore.CYAN + '%s' % self.__new_data_dir
        if not self.__create_new:
            to_ret += ' (default)'
        to_ret += '\n'

        to_ret += Fore.WHITE + 'Produce Verbose Results: \t'
        if self.__verbose:
            to_ret += Fore.GREEN + 'True'
        else:
            to_ret += Fore.RED + 'False'
        to_ret += '\n'
    
        to_ret += Fore.WHITE + 'DataFrame Dimensions: \t\t'
        if self.__df is None:
            to_ret += Fore.RED + 'data not loaded'
        else:
            to_ret += Fore.BLUE + '%dr x %dc' % self.__df.shape
        to_ret += '\n'

        pre_filter = ' (not specified)'

        to_ret += Fore.WHITE + 'Default Histogram Bins: \t' + Fore.BLUE + '%d\n' % self.__bins
        to_ret += Fore.WHITE + 'Distance Estimate: \t\t' + Fore.BLUE + '%dpc' % self.__distance_estimate
        if self.__distance_estimate == 0:
            to_ret += pre_filter
        to_ret += '\n'
        to_ret += Fore.WHITE + 'GMM Initializations: \t\t' + Fore.BLUE + '%d\n' % self.__num_inits

        to_ret += Fore.WHITE + 'Extrapolation Slopes: \t\t' + Fore.YELLOW + 'min: %.3f max: %.3f\n' \
            % (self.__slopes[0], self.__slopes[1])
        to_ret += Fore.WHITE + 'Extrapolation Intercepts: \t' + Fore.YELLOW + 'min: %.3f max: %.3f\n' \
            % (self.__intercepts[0], self.__intercepts[1])

        to_ret += Fore.WHITE + 'Members Histogram Bins: \t' + Fore.BLUE + '%d' % self.__mem_bins
        if self.__distance_estimate == 0:
            to_ret += pre_filter
        to_ret += '\n'
        to_ret += Fore.WHITE + 'Exclusion Threshold: \t\t' + Fore.BLUE + '%d' % self.__exclusion
        if self.__distance_estimate == 0:
            to_ret += pre_filter
        to_ret += '\n'
        to_ret += Fore.WHITE + 'Selection Threshold: \t\t' + Fore.BLUE + '%d' % self.__top_mem
        if self.__distance_estimate == 0:
            to_ret += pre_filter
        to_ret += '\n'

        to_ret += Fore.WHITE + 'Minimum Cluster Criterion: \t' + Fore.BLUE + '%d' % self.__min_samples
        if self.__min_samples == 0:
            to_ret += pre_filter
        to_ret += '\n'
        to_ret += Fore.WHITE + 'Epsilon Radius: \t\t' + Fore.YELLOW + '%.3f' % self.__epsilon
        if self.__min_samples == 0:
            to_ret += pre_filter
        to_ret += '\n'
        return to_ret

    
    def visualize(self,
                add_plots: dict = None) -> None:
        
        # produce figures for defaults
        fig, axs = plt.subplots(1, 2, sharey=True)
        fig.suptitle('Density Distributions of Astrometric Coordinates')
        fig.supylabel('Density')
        axs[0].set_xlabel(self.ra_lab)
        axs[0].hist(self.__df['ra'], histtype='step', color='navy', bins=self.__bins, density=True)
        axs[0].set_title('Right Ascension Density Distribution')
        axs[1].set_xlabel(self.dec_lab)
        axs[1].hist(self.__df['dec'], histtype='step', color='orangered', bins=self.__bins, density=True)
        axs[1].set_title('Declination Density Distribution')
        plt.show()
        del fig, axs

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle('Spatial Distribution of Astrometric Coordinates')
        fig.supxlabel(self.ra_lab)
        fig.supylabel(self.dec_lab)
        _, _, _, data = axs[0].hist2d(self.__df['ra'], self.__df['dec'], bins=self.__bins, density=True)
        cb = plt.colorbar(data, spacing='proportional', ax=axs[0])
        cb.set_label('Kernel Density Profile')
        axs[0].set_title(r'Density Based Spatial Distribution ($\alpha, \delta$)')
        axs[1].scatter(self.__df['ra'], self.__df['dec'], c='chocolate', marker='.')
        axs[1].set_title(r'Spatial Distribution ($\alpha, \delta$)')
        plt.show()
        del fig, axs

        # cmd
        data_tuple = self.cmd[self.__filename]
        plt.gca().invert_yaxis()
        if len(data_tuple) == 2:
            plt.scatter(self.__df[data_tuple[0]], self.__df[data_tuple[1]], c='firebrick', marker='.')
        elif len(data_tuple) == 3:
            diff = np.array(self.__df[data_tuple[0]] - self.__df[data_tuple[1]])
            plt.scatter(diff, self.__df[data_tuple[2]], c='firebrick', marker='.')
        else:
            raise KeyError(Fore.RED + 'Unrecognized key detected.')
        
        plt.xlabel(self.axes[self.__filename][0])
        plt.ylabel(self.axes[self.__filename][1])
        plt.title('%s Color Magnitude Diagram' % self.__filename.upper())
        plt.show()

        # additional plots if specified
        if add_plots is not None:
            for plot_type in add_plots.keys():
                if plot_type not in self.plots:
                    print(Fore.RED + '%s is not an acceptable plot type' % plot_type)
                    continue

                vals = add_plots[plot_type]
                for variable in vals:
                    if isinstance(variable, tuple):
                        err = False
                        for item in variable:
                            if item not in self.__df.keys():
                                print(Fore.RED + '%s is not in the %s dataframe' % (item, self.__filename))
                                err = True
                                break

                        if err:
                            continue

                        x, y = self.__df[variable[0]], self.__df[variable[1]]
                        plot = getattr(plt, plot_type)
                        if plot_type == 'hist2d':
                            tmp = pd.DataFrame(data=zip(x, y), columns=['x', 'y'])
                            tmp.dropna(inplace=True)
                            _, _, _, data = plot(tmp['x'], tmp['y'], bins=self.__bins, density=True)
                            cb = plt.colorbar(data, spacing='proportional')
                            cb.set_label('Kernel Density Profile')
                            plt.title('%s vs %s 2d Histogram' % (variable[0].upper(), variable[1].upper()))
                            del tmp
                        else:
                            plot(x, y, marker='.', c='mediumpurple')
                            plt.title('%s vs %s %s' % \
                                (variable[0].upper(), variable[1].upper(), plot_type.capitalize()))
                        
                        plt.xlabel(variable[0].capitalize())
                        plt.ylabel(variable[1].capitalize())
                        plt.show()

                    else:                                   # hist, bar, boxplot
                        if variable not in self.__df.keys():
                            print(Fore.RED + '%s is not in the %s dataframe' % (variable, self.__filename))
                            continue
                        
                        plot = getattr(plt, plot_type)
                        if plot_type == 'hist':
                            plot(self.__df[variable], bins=self.__bins, density=True, histtype='step', color='coral')
                            plt.title('%s Histogram' % variable.upper())
                            plt.ylabel('Density')
                        else:
                            tmp = pd.DataFrame(data=zip(self.__df[variable]), columns=['x'])
                            tmp.dropna(inplace=True)
                            plt.boxplot(tmp['x'], vert=False)
                            plt.title('%s Boxplot' % variable.upper())
                            plt.ylabel('%s dataset' % self.__filename)
                            del tmp                        
                        plt.xlabel(variable.upper())
                        plt.show()


    def filter_members(self, 
                    dist_est: float = 2500,
                    exclusion: int = 5,
                    top_mem: int = 25,
                    inplace: bool = True) -> Union[pd.DataFrame, None]:
        
        self.__distance_estimate = dist_est
        self.__top_mem = top_mem
        self.__exclusion = exclusion
        
        # checks
        for label in self.mem_cols:
            if label not in self.__df.keys():
                raise ValueError('%s is not present in the %d dataframe' % (label, self.__filename))
        
        # linear interpolation of Mixture Model bounds based on estimated distance
        lower_bound = self.__slopes[0] * self.__distance_estimate + self.__intercepts[0]
        upper_bound = self.__slopes[1] * self.__distance_estimate + self.__intercepts[1]

        # minimized copy for data pruning
        feeder_frame = self.__df[self.mem_cols]
        feeder_frame = feeder_frame[1000 / feeder_frame[self.mem_cols[2]] <= upper_bound]
        feeder_frame = feeder_frame[1000 / feeder_frame[self.mem_cols[2]] >= lower_bound]

        # filter members here
        feeder_vals = feeder_frame.values
        scaled_vals = pd.DataFrame(MinMaxScaler().fit_transform(feeder_vals))

        trained_gmm = self.__gmm.fit(scaled_vals)
        logits = trained_gmm.predict_proba(scaled_vals)

        if self.__verbose:
            labels = trained_gmm.predict(scaled_vals)
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].scatter(scaled_vals[0], scaled_vals[1], cmap='viridis', marker='.', c=labels)
            axs[0].set_title('Argmax Binning')
            scat = axs[1].scatter(scaled_vals[0], scaled_vals[1], cmap='plasma', marker='.', c=logits[:,1])
            axs[1].set_title('Softmax Binning')
            cb = plt.colorbar(scat, spacing='proportional', ax=axs[1])
            cb.set_label('Softmax Probability Color Bar')

            fig.suptitle('Gaussian Mixture Model Membership Binning')
            fig.supxlabel(self.pmra_lab)
            fig.supylabel(self.pmdec_lab)
            plt.show()

            del fig, axs

        bin1 = logits[:, 0]
        bin2 = logits[:, 1]
        mem1 = list(filter(lambda prob: prob >= 0.5, bin1))
        mem2 = list(filter(lambda prob: prob >= 0.5, bin2))
        ambi_pred_cnt = len(list(filter(lambda prob: 0.4 <= prob <= 0.6, bin1)))

        hist1, _ = np.histogram(bin1, self.__mem_bins)
        hist2, _ = np.histogram(bin2, self.__mem_bins)
        std1, std2 = np.std(hist1[-self.__exclusion:]), np.std(hist2[-self.__exclusion])
        probs, bins = None, 0

        if ambi_pred_cnt < .05 * len(bin1) or hist1[0] < .1 * hist2[0] or hist2[0] < .1 * hist1[0]:
            probs = bin1 if len(mem1) > len(mem2) else bin2
        else:
            probs = bin1 if std1 < std2 else bin2

        feeder_frame['probs'] = probs

        if self.__verbose:
            _, bins, _ = plt.hist(probs, bins=self.__mem_bins, histtype='step', density=True, color='sienna')
            plt.xlabel('Probability of being a Cluster Member ' + r'$[P(N_{mem})]$')
            plt.ylabel('Density of Frequency')
            plt.title('Distribution of Softmax Probabilities for Cluster Membership')
            plt.show()
        else:
            _, bins = np.histogram(probs, self.__mem_bins)
        
        self.__top_mem = min(self.__top_mem, len(bins) // 2)
        feeder_frame = feeder_frame[feeder_frame['probs'] >= bins[-self.__top_mem]]
        feeder_frame.drop(columns=['probs'], inplace=True)

        pmra_maps = np.array(feeder_frame[self.mem_cols[0]])
        filtered_df = deepcopy(self.__df)
        filtered_df = filtered_df[filtered_df[self.mem_cols[0]].isin(pmra_maps)]
        
        if self.__create_new:
            if not path.exists(self.__new_data_dir):
                mkdir(self.__new_data_dir)

            new_file = ''
            if path.exists('%s/%s' % (self.__new_data_dir, self.__filename)):
                new_file = '%s/%s_1.%s' % (self.__new_data_dir, self.__filename, self.__extension)
            else:
                new_file = '%s/%s.%s' % (self.__new_data_dir, self.__filename, self.__extension)
            filtered_df.to_csv(new_file, index=False)
        
        if inplace:
            self.__df = filtered_df
            return None
        
        return filtered_df


    def dbscan_filter(self,
                    n_mem: int = 10,
                    inplace: bool = True) -> Union[pd.DataFrame, None]:
        
        # checks
        if n_mem < 2:
            raise ValueError(Fore.RED + 'There must be at least two member in a cluster')
        for field in self.mem_cols:
            if field not in self.__df.keys():
                raise ValueError(Fore.RED + '%s is not present in %s dataframe' % (field, self.__filename))
        
        self.__min_samples = n_mem

        x = np.array(self.__df[self.mem_cols[0]])
        y = np.array(self.__df[self.mem_cols[1]])
        xy = np.vstack([x, y]).T
        X = StandardScaler().fit_transform(xy)          # standardize data

        neighbors = NearestNeighbors(n_neighbors=self.__min_samples)
        neighbor_fit = neighbors.fit(X)
        distances, _ = neighbor_fit.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        i = np.arange(len(distances))
        kneedle = KneeLocator(i, distances, S=1, curve='convex', direction='increasing')

        if self.__verbose:
            plt.plot(distances)
            plt.axvline(kneedle.knee, color='crimson', linestyle='--', label='Elbow')
            plt.legend(loc='best')
            plt.xlabel('Datapoints')
            plt.ylabel(r'$\epsilon$')
            plt.title('Elbow Estimate for DBSCAN')
            plt.show()

        self.__epsilon = kneedle.knee_y
        self.__dbscan = DBSCAN(eps=self.__epsilon, min_samples=self.__min_samples)
        db_fit = self.__dbscan.fit(X)
        core_samples_mask = np.zeros_like(db_fit.labels_, dtype=bool)
        core_samples_mask[db_fit.core_sample_indices_] = True
        c_labels = db_fit.labels_

        if self.__verbose:
            plt.scatter(x, y, c='indigo', marker='.', label='Original Data')
            plt.xlabel(r'$\mu_{\alpha*} cos(\delta)$' + ' (mas/yr)')
            plt.ylabel(r'$\mu_{\delta}$' + ' (mas/yr)')
            plt.title('Vector Point Diagram with DBSCAN selection')
            plt.scatter(x[c_labels != -1], y[c_labels != -1], marker='.', c='salmon', label='DBSCAN selection')
            plt.legend(loc='best')
            plt.show()

        filtered_param = x[c_labels != -1]
        filtered_df = self.__df[self.__df[self.mem_cols[0]].isin(filtered_param)]

        if self.__create_new:
            if not path.exists(self.__new_data_dir):
                mkdir(self.__new_data_dir)

            new_file = ''
            if path.exists('%s/%s' % (self.__new_data_dir, self.__filename)):
                new_file = '%s/%s_1.%s' % (self.__new_data_dir, self.__filename, self.__extension)
            else:
                new_file = '%s/%s.%s' % (self.__new_data_dir, self.__filename, self.__extension)
            filtered_df.to_csv(new_file, index=False)

        if inplace:
            self.__df = filtered_df
            return None
        
        return filtered_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Append MetaData for working with the Cluster Class')

    parser.add_argument('data_dir', type=str, default='data', help='The directory to read in the data from')
    parser.add_argument('data', type=str, default='gaia.csv', help='File name (with extension) to extract data from')
    parser.add_argument('-n', '--new_csv', action='store_true', help='create a new csv post filtering')
    parser.add_argument('-R', '--results_dir', type=str, default='.', help='The directory to store the new data in')
    parser.add_argument('-V', '--verbose', action='store_true', help='produce figures for utility functions')
    parser.add_argument('-i', '--inits', type=int, default='1', help='Number of initializations for GMM')
    parser.add_argument('-p', '--inplace', action='store_true', help='Performs filtering in place')
    parser.add_argument('-e', '--dist_est', type=float, default=2500,\
        help='Best estimate for distance of the cluster in parsecs', required=True)
    parser.add_argument('-E', '--exclusion', type=int, default=5,\
        help='Exclusion Threshold for Standard Deviation calculations')
    parser.add_argument('-m', '--top_m', type=int, default=25, help='Selection threshold for filtered output')
    parser.add_argument('-M', '--mem', type=int, default=10,\
        help='Minimum number of points to be considered a cluster')

    args = parser.parse_args()

    clust = Cluster(dataset_path='%s/%s' % (args.data_dir, args.data),
                    num_inits=args.inits,
                    new_csv=args.new_csv,
                    new_dir=args.results_dir,
                    verbose=args.verbose)
    res = clust.load_data(inplace=args.inplace)
    res2 = clust.filter_members(dist_est=args.dist_est,
                                exclusion=args.exclusion,
                                top_mem=args.top_m,
                                inplace=args.inplace)
    res3 = clust.dbscan_filter(n_mem=args.mem,
                            inplace=args.inplace)
    print(clust)

    # manually edit these ->
    clust.visualize(dict(scatter=[('pmra', 'pmdec')], boxplot=['pmra', 'pmdec']))
