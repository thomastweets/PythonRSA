# -*- coding: utf-8 -*-

################################################################################
################# Representational Similarity Analysis #########################
################################################################################

# Amelie Haugg
# Julia Brehm
# Pia Schröder

import os
from os.path import dirname, abspath
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import markdown
import webbrowser

# Global variables and their default values

matrix_plot1 = True
matrix_plot2 = False
bar_plot = False
correlations1 = False
correlations2 = False
pvalues = False
no_relabelings = 10000
dist_metric = 1
output_first = True
output_second = False
scale_to_max = False

now = datetime.datetime.now()


def import_data(paths):
    """ Import header and data matrix from VOM files specified in paths. Returns
    dictionary DATA containing data set names as keys."""

    DATA = dict()

    # Iterate through files and save data in dictionary
    for set_no, file_name in enumerate(paths):

        header = dict()

        # Read header and store last line
        with open(file_name, 'r') as file:

            for index, line in enumerate(file):
                string_list = line.split()
                item_list = [int(i) if i.isdigit() else i for i in string_list]

                # For non empty list save first element as key and rest as value in
                # header dictionary
                if item_list:
                    key = item_list.pop(0)
                    if len(item_list) > 1:
                        header[key] = item_list
                    else: header[key] = item_list.pop()
                # Use 'NrOfVoxels' as indicator for the end of the header
                if 'NrOfVoxels:' in line:
                    break

        header_end = index + 1
        # Read data into array
        data = np.loadtxt(file_name, skiprows = header_end)

        # Save data set in DATA dictionary
        key = "data_set_" + str(set_no + 1)
        DATA[key] = {'header': header, 'data': data}

    return DATA


def extract_data(DATA):
    """ Get voxel data from data matrices in DATA. One matrix per area, rows = voxels,
    columns = conditions. """
    # Extracts those columns in data that contain measurements (excluding voxel coordinates)
    data = []
    for i in range(1,len(DATA)+1):
        data.append(DATA['data_set_' + str(i)]['data'][:,3:])
    return data


def first_order_rdm(condition_data):
    """ Return Specified distance matrices (1 = Pearson correlation,
    2 = Euclidian distance, 3 = Absolute activation difference)  of data in
    input matrices. One matrix per area/subject/method/...
    Number of rows/columns = number of conditions = number of columns in each
    matrix in condition_data"""

    RDMs = list()

    # Iterate through matrices in condition_data and save one RDM per matrix
    for i in range(len(condition_data)):

        if dist_metric == 1:
            # Use correlation distance
            RDM = 1-np.corrcoef(condition_data[i],rowvar=0)

        elif dist_metric == 2:
            # Use Eucledian distance
            RDM = cdist(condition_data[i].T,condition_data[i].T,'euclidean')

        elif dist_metric == 3:
            # Use absolute activation difference
            means = np.mean(condition_data[i], axis=0) # Determine mean activation per condition
            m, n = np.meshgrid(means,means) # Create all possible combinations
            RDM = abs(m-n) # Calculate difference for each combination

        RDMs.append(RDM)

    return RDMs


def get_pvalue(matrix1, matrix2):
    """ Randomize condition labels to test significance """

    order = range(0,len(matrix2))
    dist = np.zeros(no_relabelings)

    # First, determine actual correlation
    flat1 = matrix1.flatten(1).transpose()
    flat2 = matrix2.flatten(1).transpose()
    corr = spearmanr(flat1,flat2)[0]

    # Relabel N times to obtain distribution of correlations
    for i in range(0,no_relabelings):
        np.random.shuffle(order)
        dummy = matrix2.take(order, axis=1).take(order, axis=0)
        flat2 = dummy.flatten(1).transpose()
        dist[i] = spearmanr(flat1,flat2)[0]

    # Determine p value of actual correlation from distribution
    p = float((dist >= corr).sum()) / len(dist)

    # Mit dieser Methode braucht man mindestens 4 conditions, also 4!=24 mögliche
    # Reihenfolgen um auf p < 0.05 zu kommen. Nicht gut!

    return p

def bootstrap(data):
    """ computes the variability of the obtained second-order RDM (i.e. distance
    between areas, models, ...) for the same experiment with different stimuli
    by bootstrapping 100 times from the condition set. """

    all_RDMs = list()

    # Iterate through 100 resamplings
    for ind in range(100):
        index = np.random.random_integers(0, high=len(data[0].T)-1, size=(1,len(data[0].T)))[0]
        new_data = np.array(data)
        # Reorder columns in data (conditions)
        for elem in range(len(data)):
            new_data[elem] = new_data[elem][:,index]
        # Recompute first and second-order RDMs with new conditions
        new_RDM1 = first_order_rdm(list(new_data))
        new_RDM2 = second_order_rdm(new_RDM1, data, False)[0]
        # Remove off-diagonal zeros to avoid artefactually small standard deviations
        m_index = [new_RDM2 == 0]
        ident = np.invert(np.identity(len(new_RDM2), dtype=bool))
        m_index = m_index & ident
        new_RDM2[m_index[0]] = np.nan

        all_RDMs.append(new_RDM2)

    all_RDMs = np.array(all_RDMs)

    # Compute standard deviation along first dimension (across RDMs)
    variability = np.nanstd(all_RDMs,0)

    return variability

def second_order_rdm(RDMs, data, firstcall):
    """ Returns representational dissimilarity matrix computed with Spearman rank correlations
    between variable number of equally sized input matrices. """

    # Flatten input matrices
    flat = [m.flatten(1) for m in RDMs]


    flat = np.array(flat).transpose()

    # Compute Spearman rank correlation matrix
    c_matrix = spearmanr(flat)[0]

    # In case only two conditions are compared, spearmanr returns single correlation
    # coefficient and c_matrix has to be built manually
    if not(isinstance(c_matrix, np.ndarray)):
        c_matrix = np.array([[1,c_matrix],[c_matrix,1]])

    # Compute RDM (distance matrix) with correlation distance: 1 - correlation
    RDM = np.ones(c_matrix.shape) - c_matrix

    p_values = []
    variability = []

    if firstcall:

        if bar_plot and len(RDMs)>2:

            # Determine variability of distance estimates for different stimuli
            # Bootstrap from condition set (100 times, with replacement)
            variability = bootstrap(data)

        if pvalues or bar_plot:
            # Determine significance of second order RDM
            p_values = np.zeros(RDM.shape)

            # Iterate through pvalue matrix and fill in p-values but only for upper
            # triangle to improve performance
            for i in range(0,len(p_values)):
                for j in range(i,len(p_values)):
                    p_values[i,j] = get_pvalue(RDMs[i], RDMs[j])

            # mirror matrix to obtain all p-values
            p_values = p_values + np.triu(p_values,1).T


    return [RDM, p_values, variability]


def plot_RDM(RDMs, labels, names, fig):
    """ Create RDM plot. Creates one first-order plot for each area if fig=1
    and a single second-order plot if fig=2."""

    # Determine optimal arrangement for plots
    rows = int(np.sqrt(len(RDMs)))
    columns = int(np.ceil(len(RDMs)/float(rows)))

    ticks = np.arange(len(labels))

    # Use maximum value in RDMs for scaling if desired
    dist_max = np.max(np.array(RDMs))

    if fig == 1:
        f = plt.figure(fig, figsize=(18, 8))
    if fig == 2:
        f = plt.figure(fig, figsize=(6, 6))

    # New: add_subplot instead of subplots to control figure instance
    for index in np.arange(len(RDMs)):

        ax = f.add_subplot(rows,columns,index+1, xticklabels = labels, yticklabels = labels, xticks = ticks, yticks = ticks)

        if scale_to_max:
            im = ax.imshow(RDMs[index], interpolation = 'none', cmap = 'jet', vmin = 0, vmax = dist_max)
        else:
            im = ax.imshow(RDMs[index], interpolation = 'none', cmap = 'jet')

        for label in ax.get_xticklabels():
            label.set_fontsize(6)

        ax.xaxis.tick_top()
        ax.set_title(names[index], y = 1.08)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        dist_max = np.max(RDMs[index])
        cbar = plt.colorbar(im, ticks=[0, dist_max], cax=cax)
        cbar.ax.set_yticklabels(['0', str(np.around(dist_max,decimals=2))])

    cbar.ax.set_ylabel('Dissimilarity')

    f.subplots_adjust(hspace=0.1, wspace=0.3)


    if fig == 1:
        if dist_metric == 1:
            f.suptitle('First order distance metric: Correlation distance', y=0.9, fontsize=18)
        elif dist_metric == 2:
            f.suptitle('First order distance metric: Euclidean distance', y=0.9, fontsize=18)
        elif dist_metric == 3:
            f.suptitle('First order distance metric: Absolute activation difference', y=0.9, fontsize=18)

    figure_name = "Figure%d_%d-%d-%d-%d-%d-%d.png" % (fig, now.day, now.month, now.year, now.hour,
               now.minute, now.second)
    plt.savefig(figure_name, transparent=True)

    return figure_name


def plot_bars(RDM, pvalues, variability, names):
    """ Creates bar plot depicticting the distances between different areas.
    Bars are sorted by significance, errorbars indicate the standard error of
    the distnace estimate (estimated as the standard deviation of 100 distance
    estimates obtained from bootstrapping of the condition labels)"""

    length = len(RDM)

    f = plt.figure(3, figsize=(14,6))

    for index in np.arange(length):

        maxim = np.max(RDM[index])

        xticks = np.arange(length-1)+1

        d_values = RDM[index,:]
        plot_dvalues = d_values[d_values != 0]

        v_values = variability[index,:]
        plot_vvalues = v_values[d_values != 0]

        p_values = pvalues[index,:]
        plot_pvalues = np.around(p_values[d_values != 0], decimals=4)

        plot_names = np.array(names)[d_values != 0]

        sort = np.argsort(plot_pvalues)

        ax = f.add_subplot(1,length, index+1, xticks = xticks, xticklabels = plot_pvalues[sort])
        ax.set_ylabel('Correlation distance (1-Spearman rank correlation)')
        ax.set_xlabel('P-values')

        ax.bar(xticks, plot_dvalues[sort], 0.5, yerr = plot_vvalues[sort], error_kw=dict(ecolor='black', lw=2), align = 'center')
        scale_y = max(plot_dvalues + plot_vvalues)+maxim*0.1
        plt.axis([0.5, length-0.5, 0, scale_y])

        ax.set_title(names[index])

        for ind in np.arange(length-1):
            ax.text(xticks[ind], scale_y*0.1, plot_names[sort][ind],
                 rotation='vertical', horizontalalignment='center',
                 backgroundcolor='w', color='k', visible=True)


    f.subplots_adjust(hspace=0.1, wspace=0.3)
    figure_name = "Figure3_%d-%d-%d-%d-%d-%d.png" % (now.day, now.month, now.year, now.hour,
               now.minute, now.second)
    plt.savefig(figure_name, transparent=True)

    return figure_name


def generate_output(*args):
    """ Generates text file including all output and converts it into html
    (markdown) file """

    if len(args) > 3:
        [withinRDMs, betweenRDM, names, labels] = args
    else:
        [withinRDMs, names, labels] = args

    # Produce text file
    filename = "RSA_output_%d-%d-%d-%d-%d-%d.txt" % (now.day, now.month, now.year, now.hour,
               now.minute, now.second)

    output = "RSA_output_%d-%d-%d-%d-%d-%d.html" % (now.day, now.month, now.year, now.hour,
               now.minute, now.second)

    with open(filename, 'w') as fid:

        fid.write("#Representational similarity analysis\n\n")
        fid.write("###Areas: "+str(', '.join(names))+"\n")
        fid.write("###Conditions: "+str(', '.join(labels))+"\n\n\n\n")

        # first-order RDMs
        if output_first:

            fid.write("##First-order analysis\n\n")

            # Numerical correlations
            if correlations1:

                distances = {1:'Correlation distance', 2:'Euclidean distance', 3:'Absolute activation difference'}

                fid.write("###Dissimilarity between conditions: "+distances[dist_metric]+"\n\n")
                for ind in np.arange(len(withinRDMs)):
                    fid.write("\n###"+names[ind]+"\n")
                    np.savetxt(fid, withinRDMs[ind], fmt='%.4f')# , header="\n"+names[ind]+"\n")
                    fid.write("\n")

            # RDM Plot
            if matrix_plot1:
                figure_name = plot_RDM(withinRDMs, labels, names, 1)
                fid.write("![Figure1](%s)" % figure_name)

        # second-order RDM
        if output_second:

            fid.write("\n")
            fid.write("##Second-order analysis\n\n")

            # Numerical correlations
            if correlations2:

                fid.write("###Dissimilarity between areas: 1-Spearman rank correlation\n\n")
                np.savetxt(fid, betweenRDM[0], fmt='%.4f')
                fid.write("\n\n")

            # P-values
            if pvalues:

                fid.write("###Statistical significance of Dissimilarity between areas\n")
                fid.write("P-values are obtained by random relabeling of conditions.\nNo. of relabelings = %d \n\n" % (no_relabelings))
                np.savetxt(fid, betweenRDM[1], fmt='%.4f')
                fid.write("\n\n")

            # RDM plot
            if matrix_plot2:
                figure_name = plot_RDM([betweenRDM[0]], names, ['Second order RDM'], 2)
                fid.write("\n")
                fid.write("![Figure2](%s)" % figure_name)
                fid.write("\n")

            # Bar plot
            if bar_plot and len(betweenRDM[0])>2:

                figure_name = plot_bars(betweenRDM[0], betweenRDM[1], betweenRDM[2], names)
                fid.write("\n")
                fid.write("![Figure3](%s)" % figure_name)
                fid.write("\n")

    with open(output, 'w') as output_file:

        html = markdown.markdownFromFile(filename, output_file, extensions=['markdown.extensions.nl2br'])

    os.remove(filename)
    webbrowser.open(output, new=2)


def RSA(paths, files, labels):
    ''' Imports input files, extracts relevant data, computes first and second
    order RDMs and plots them'''

    data = import_data(paths)
    data = extract_data(data)
    withinRDMs = first_order_rdm(data)

    names = [file[0:-4] for file in files]

    if output_second:
        betweenRDM = second_order_rdm(withinRDMs, data, True)

    if output_second:
        generate_output(withinRDMs, betweenRDM, names, labels)
    else:
        generate_output(withinRDMs, names, labels)
