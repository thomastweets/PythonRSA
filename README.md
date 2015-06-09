# RSA Python Documentation

RSA Python is a program to perform a representational similarity analysis on functional magnetic resonance imaging data.  
The analysis is following the principles described in the paper 'Representational Similarity Analysis - Connecting the Branches of Systems Neuroscience' by Nikolaus Kriegeskorte, Marieke Mur and Peter Bandettini (2008).  
It is the result of a project work at Maastricht University by Pia Schroeder, Amelie Haugg and Julia Brehm under the supervision of Thomas Emmerling.

1. Representational Similarity Analysis
2. File Format
3. Condition List
4. Options
 1. First Order Analysis
  * Settings
 2. Second Order Analysis
  * Settings

## Representational Similarity Analysis

In RSA data is analyzed and compared based on the informational content. That way data from different modalities (such as different brain areas, models and methods) can be more easily compared. The analysis is described in detail in Kriegeskorte, Mur and Bandettini (2008).

## File Format

RSA Python is compatible with vom-file output from the [BrainVoyager](http://www.brainvoyager.com/) software.
It currently does not support any other file format.

## Condition List

Insert your condition labels separated by a space. The order should be the same as in the vom-files (i.e. in the order of the columns).

## Options
### First Order Analysis

In the first order analysis the specified distance between columns within each selected vom-file is computed (See 'Distance metric'). The higher the value, the bigger the difference between two conditions.  
The first-order analysis option is selected by default and will always be carried out even if unchecked. Only if checked, further options can be selected.
The program will always produce a html-file containing the results of the analysis asked for.

#### Settings

##### Matrix Plots

This option plots distance matrices separately for each file and save the plots in one png-file and additionally in the output file. The plot is color coded scaled to the maximum value in each matrix separately and indicates dissimilarity between conditions. The matrix dimensions will match the number of conditions and is symmetric along the diagonal.

##### Correlations

This option will include the plain distance matrices in the output file.

##### Scale to max

Checking this option will scale the color coding of the output plots to the maximum value in all distance matrices. This increases comparability across output plots.

##### Distance metric

Here, different options to compute the dissimilarity between conditions can be selected.

* Correlation distance: 1 - Pearson' product-moment correlation coefficient
* [Euclidian distance](http://en.wikipedia.org/wiki/Euclidean_distance)
* Absolute activation distance: mean absolute difference in activation

### Second Order Analysis

This option can only be selected if two or more files are selected. In the second order analysis, the dissimilarity between the distance matrices from the first-order analysis is computed. This is done using the following formula: 1 - Spearman's rank correlation. By default, p-values are obtained by randomly reordering rows and columns of the matrices 10000 times and computing how often this distance is elicited by chance.

#### Settings

##### Matrix Plots

When checked, this option will plot and save the matrix the same way as in the first-order analysis, using the single distance matrix obtained in the second-order analysis. The matrix dimensions will match the number of files used as input and is symmetric along the diagonal as well.

##### Bar Plots

This option will plot and save one bar plot for each file, showing distances and p-values to all other files.

##### Correlations

Checking this option will include the distance matrix in the output file.

##### p-values

This option includes a list of p-values for each distance value in the output file.

##### No. of Permutations

The number of permutations for significance testing can be changed. This option is limited to 100,000 permutations.
