# S3D_application
Example codes of the workflow of S3D for folded proteins and IDPs (Intrinsically Disordered Proteins)

This repository provides the example codes of the whole workflow of S3D (Structure and State Space Decomposition). We use Villin as the example for fast-folding proteins and ACTR for IDPs.

The workflow for fast-folding proteins contains:
* TICA (Time-lagged Independent Component Analysis) for dimensionality reduction
* MSM (Markov State Model) constructed by performing k-means clustering in the TICA space
* HMM (Hidden Markov Model) estimation to extract metastable states
* Space time diffusion map to identify coherent domains for each metastable state
* Obtain assembly units as the intersections of coherent domians from differernt metastable states

The workflow for IDPs contains:
* Space time diffusion map to identify coherent domains for each simulation
* Obtain assembly units as the intersections of coherent domians from differernt simulations
