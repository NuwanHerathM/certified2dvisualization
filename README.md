# Fast curve visualization


Visualization of a implicit curve defined by a bivariate polynomial on the square [-1,1] * [-1,1] with a given resolution n.
<p align="center">
  <img src="images/unit_circle_128_sub_intvl.png" width="200">
</p>

Two methods are implemented:
- multipoint evaluation + subdivision
- multipoint evaluation + Taylor approximation

## Prerequisites

### A standard Python package

- `scipy`

### Some non-standard Python packages

- `flint-py` (available for python <= 3.8 with pip for now (July 2021))
- `codetiming`
- `pyinterval`

Note that it is `flint-py`, not `python-flint`.

Not used currently:
- `binarytree`
- `vispy` ([here](https://vispy.org/installation.html))
- `numba` (requires `llvmlite` and thus `llvm-10` or `llvm-9`)

### Third-party program (not used currently)

Download the binary file corresponding to your platform from [ANewDsc's website](http://anewdsc.mpi-inf.mpg.de/).\
Rename `test_descartes_os64` as `test_descartes` or use a symbolic link. Add the directory in which `test_descartes` is to your PATH, if it is not already the case.

## Running the program

Here are some examples of commands to use the code.

### Basic program

Using the default polynomial (a circle of radius 1):

```
python3 main.py 64
```

Using another polynomial:

```
python3 main.py 512 -poly ../polys/random.poly
```

### Subdivision or Taylor approximation (with edge enclosing)

Using subdivision (nothing to do):

```
python3 main.py 64
```

Using Taylor approximation:

```
python3 main.py 64 -taylor
```

### Subdivison (with pixel enclosing)

Using interval arithmetic (nothing to do):

```
python3 certipixel.py 64
```

## Help

For additional information use:

```
python3 main.py --help
```
