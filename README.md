# Multilinear Regression with Interrelated Lines

In some instances, it can be interesting to analyze multiple simultaneous linear behaviours in a same pool of datapoints. This is henceforth called a multilinear regression. The presented script aims at doing precisely that: for a pool of i datapoints (x_i,y_i), it fits it with a set of n interrelated lines. That is, all the lines follow the same structure and same base parameters, but they are different among themselves by an integer factor. Examples are given below to make this definition clearer. It is naturally also possible to perform a free multilinear regression, where the various lines being fitted are completely independent, however that's not the scope of the scrip here presented.

When can it be useful to keep the lines interrelated? In the case of completely independent lines, the number of independent parameters to be determined increases quite significantly as we increase the number of lines, which makes the computation heavier, but also reduces the robustness of the fit, as the ratio between data points and free variables plummets the more lines we fit. If the lines are interrelated, we have a more compact and simple fit. Physically, that can also have interesting analyses, but that's outside of the scope of discussion of this text.

A mathematical description of the fitting method is present in //cite here proton paper//.

Two practical examples of application of the script:

- We wish to fit the data with 4 lines (n=4) of the type y=k(ax+b)+c, where k is an integer from [0,1,2,3] and a, b and c are the free parameters that we want to find. There will be four lines: y=c, y=ax+b+c, y=2(ax+b)+c, y=3(ax+3)+c, however only three parameters to determine. If each line had two or three free parameters, for instance, this would lead to eight or twelve parameters to determine. 

- We wish to fit the data with 6 lines (n=6) of the type y=k[exp(ax^2+bx+c)]+d, where k is once again an integer from [0,1,2,3,4,5]. Now we have four parameters a, b, c, d. Six lines: y=d, y=exp(ax^2+bx+c)+d, ..., y=6[exp(ax^2+bx+c)]+d. Only four parameters to determine. If each line had four free parameters, this would lead to twenty four parameters to determine.

# Structure of the Repository

This repository has three files: 

- *README.md*

- *MultilinearGeneralFit.py*: full script featuring:
      > Multilinear regression to determine the optimal parameters (minimum square error approach)
      > Clone the original data file and add to it the model values, errors, and more
      > Plot various information of interest:
            . Optimal parameter model + Data Points
            . Deviations from the model
            . Distribution of deviations from the model + Normal fit
            . "Collapsed" fit (transform multilinear into the equivalent linear fit conserving variance)
            . Bayesian fit (adapted from https://github.com/krasserm/bayesian-machine-learning)
            
- *MultilinearPlotter.py*: shortened script, equivalent to MultilinearGeneralFit.py, but without calculating optimal parameters. Useful if this has been calculated and only plotting is of interest.

- *Data.txt*: an example data file - it is important that the used data file has this format, so that the data retrieval works - or else, the code might need to be changed to accomodate for that

- *Alternative Fits* folder: a folder with alternative fitting types. Currently featuring only Minimax approach:
      > *MultilineFit_Minimax.py*: A script to calculate the optimal parameters to fit the data using a minimax approach (minimizing the maximum deviation) and plot it.


