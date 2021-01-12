# splineBART

Eventually will contain an R package for fitting a sum-of-treed splines model.
Basically it's BART but instead of having each tree output a scalar, the trees now output a vector of coefficients to be used in a B-splines basis expansions.

This will essentially be a sum-of-trees extension of the paper [``A Bayesian tree approach to identify the effect of nanoparticles' properties on toxicity profiles''](https://projecteuclid.org/euclid.aoas/1430226097) by Low-Kam et al.

Eventually you will be able to install the R package from this repository but much busywork remains to be done (i.e. handling the exports, filling out the `DESCRIPTION` and `NAMESPACE` files, checking that the build works, etc.).


The directory `test` contains some scripts used for testing and development.
