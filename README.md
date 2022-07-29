# xy-kt
## Monte Carlo simulation of the 2D XY model and the Kosterlitz-Thouless transition
### Features:
- Real-time interactive simulation of 2D classical XY model on a square lattice using `matplotlib`
- Real-time visualisation of vorticity and vortex-antivortex pair creation and annihilation
- Computation of macroscopic variables to demonstrate [KT transition](https://en.wikipedia.org/wiki/Kosterlitz%E2%80%93Thouless_transition)
- Support for single-flip Metropolis algorithm (thermalizes faster on average) and cluster-flip Wolff algorithm (avoids critical slowing down by reducing autocorrelation) 
- Usage of `numba` JIT compilation and `joblib` parallelization for extra speed

### Getting Started:
```
> git clone https://github.com/nkarve/xy-kt.git
> cd src
> python xy_realtime.py
```

<img src="/demos/rt.gif">
