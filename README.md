# Learning Stochastic Multiscale Models

<p align="center">
  <img align="middle" src="./images/cylinder_2d/512_5_2000_0.0001_2000_augment/test_multiscale.gif" alt="Multiscale 2D Cylinder Flow Model" width="100%"/>
</p>

If the above gif is not rendering, view it [here.](images/cylinder_2d/512_5_2000_0.0001_2000_augment/test_multiscale.gif)

This repository accompanies the preprint "Learning Stochastic Multiscale Models" available [here](). It contains the training scripts and results for all test cases presented therein. Dependencies are handled with [Poetry](https://github.com/python-poetry/poetry) and listed in `pyproject.toml`. If you use our work, please cite:

`
testing 123
`

Certain physical systems, such as fluid flow at high Reynolds number, require the dynamics to be resolved across a large continuum of length and time scales. This approach, like closure modelling, imposes a scale separation between directly-resolved macroscale features and unresolved microscale features. Unlike closure modelling, which seeks to model the system purely in terms of the macroscale features, we augment it with a microscale component. This component comes from encoding the residual between the training data and the macroscale representation. We then model the dynamics as a coupled pair of differential equations, allowing for efficient inference and prediction.
