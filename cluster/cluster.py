import pandas as pd
import numpy as np
from os import path
from colorama import init, Fore
from copy import deepcopy
from matplotlib import pyplot as plt
from typing import Union

# system settings
import sys

sys.tracebacklimit = 5
init(convert=True, autoreset=True)


class Cluster(object):

    def __init__(self,
                dataset_path: str, 
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
        
        
        # meta data for new file
        self.__create_new = new_csv
        self.__new_data_dir = new_dir if self.__create_new else '%s (default)' % self.__data_directory

        # other meta data
        self.__verbose = verbose
        self.__df = None
        self.__bins = 45

        self.ra_lab = r'Right Ascension (deg) [$\alpha$]'
        self.dec_lab = r'Declination (deg) [$\delta$]'
        self.axes = dict(gaia=(r'$G-R_P$', r'$G$'), sdss=(r'$G-R$', r'$G$'), simbad=(r'$B-V$', r'$B$'))
        self.cmd = dict(gaia=('g_rp', 'g'), sdss=('g','r','g'), simbad=('B','V','B'))
        self.plots = ['hist', 'hist2d', 'plot', 'scatter', 'boxplot']

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
        
        to_ret += Fore.WHITE + 'Data Output Directory: \t\t' + Fore.CYAN + '%s\n' % self.__new_data_dir
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

        to_ret += Fore.WHITE + 'Default Histogram Bins: \t' + Fore.BLUE + '%d\n' % self.__bins

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


if __name__ == '__main__':
    clust = Cluster('raw_data/gaia.csv', True, 'new')
    clust.load_data()
    # print(clust)
    clust.visualize(dict(scatter=[('pmra', 'pmdec')], boxplot=['pmra', 'pmdec']))