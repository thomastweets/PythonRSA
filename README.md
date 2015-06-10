# RSA Python Documentation

RSA Python performs representational similarity analysis (RSA) on functional magnetic resonance imaging (fMRI) data.  
The analysis follows the principles described in ['Representational Similarity Analysis - Connecting the Branches of Systems Neuroscience'](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2605405/) by Nikolaus Kriegeskorte, Marieke Mur and Peter Bandettini (2008).  
It is the result of a project work at the Department of Cognitive Neuroscience, Maastricht University by Pia Schroeder, Amelie Haugg and Julia Brehm under the supervision of Thomas Emmerling.

1. Representational Similarity Analysis
2. Software Requirements
3. File Format
4. Condition List
5. Options
 1. First Order Analysis
 2. Second Order Analysis
6. Output

## Representational Similarity Analysis

RSA uses activity patterns in fMRI data to compare representations in different stimulus conditions and modalities (e.g., different imaging techniques). It allows abstracting from the actual measurements to representational dissimilarity matrices (RDMs), which can be used to infer the relation between different modalities, models, brain areas, subjects, and species. As such it overcomes correspondence problems that are commonly encountered in systems neuroscience. The analysis is described in detail in [Kriegeskorte, Mur and Bandettini (2008)](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2605405/).

## Software Requirements

Running this software requires [Python 2.7](https://www.python.org/download/releases/2.7/) including the following packages:

* [wxPython](http://www.wxpython.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Markdown](https://pypi.python.org/pypi/Markdown)

## File Format

RSA Python is compatible with fMRI data in vom-file format from the [BrainVoyager](http://www.brainvoyager.com/) software.
It currently does not support any other file format.
In order to perform second-order analysis, upload more than one file (each file corresponding to a set of measurements in for example a different brain area). Note that each set of measurements must contain the same set of conditions in the same order.
File names should be meaningful with respect to the data they contain, as they are used for naming in the analysis output.

## Condition List

Insert your condition labels separated by a space. The order should be the same as in the vom-files (i.e. in the order of the columns containing data for each condition).

## Options
### First Order Analysis

In first order analysis the distance between conditions is computed, with higher values corresponding to larger differences between conditions. The specific distance metric is specified in 'Distance metric' (possible options are correlation distance, Euclidian distance, and absolute activation difference). Optional outputs include numerical output of the computed distances as well as a plot of the first-order RDM. Choose 'scale to max' if you would like to use the same scale for all first-order RDMs.

#### Settings

##### Matrix Plots

This option plots first-order RDMs for each file, indicating dissimilarities between conditions. The plot is color-coded with red colors corresponding to large distances and blue colors corresponding to small distances. Each plot is scaled to its maximum value. The matrix dimensions match the number of conditions and the RDMs are symmetric along the diagonal.

##### Correlations

This option outputs the distances between conditions in numerical form.

##### Scale to max

Use this option to use the same scale for all RDMs (the overall maximum distance value across RDMs is used). This increases comparability across RDMs.

##### Distance metric

Chose between different options to compute the dissimilarity between conditions:

* Correlation distance: 1 - Pearson' product-moment correlation coefficient
* [Euclidian distance](http://en.wikipedia.org/wiki/Euclidean_distance)
* Absolute activation distance: mean absolute difference in activation

### Second Order Analysis

This option can only be selected if two or more files are selected. In second order analysis, the dissimilarity between first-order distance matrices is computed to allow comparison between modalities or brain areas. The resulting RDM uses correlation distance (1 - Spearman's rank correlation) as distance metric. P-values indicating the significance of the computed correlations are obtained by random relabeling of condition labels. This is a conservative estimate with the smallest possible p-value being (1/number of permutations). The variation of the distance estimates that is expected if the experiment were repeated with different stimuli, is computed by bootstrapping 100 times from the condition set (note that this estimate becomes meaningful only if a large number of different conditions is used).

#### Settings

##### Matrix Plots

This option will plot the second-order RDM representing differences between the data contained in the input files (for interpretation see first-order analysis). The matrix dimensions match the number of files used as input and the obtained RDM is symmetric along the diagonal.

##### Bar Plots

This option outputs a bar plot, showing distances between first-order RDMs. The Bars are sorted by p-values. Error bars represent the variability of the distance estimates if the experiments were repeated with different stimuli.

##### Correlations

This option outputs the correlation distance in numerical form.

##### P-values

This option the p-values of the distance estimates in numerical form.

##### No. of Permutations

Use this option to adjust the number of permutations used for estimating the statistical significance of the correlation distances (100-100000). The larger the number of permutations, the more reliable the estimate will be. Note however, that a large number of permutations is computationally demanding and will cause the program to notably slow down.

## Output

The analysis outputs and saves an html file including all numerical output and plots specified in *Options*. In addition, all created plots are saved separately as png-files. Note that all outputs are saved in the current working directory and are automatically named based on the current date and time. We advise to sort and rename the files after each analysis to avoid confusion.
