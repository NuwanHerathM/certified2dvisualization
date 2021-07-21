# Title

This is beginning to become an actual readme...

## Prerequisites

### Some non-standart Python packages

- `flint-py` (available for python <= 3.8 with pip for now (July 2021))
- `codetiming`
- `binarytree`
- `vispy` ([here](https://vispy.org/installation.html))
- `pyinterval`
Not used currently:
- `numba` (requires `llvmlite` and thus `llvm-10` or `llvm-9`)

### Third-party programm

Download the binary file corresponding to your platform from [ANewDsc's website](http://anewdsc.mpi-inf.mpg.de/).
Rename `test_descartes_os64` as `test_descartes` or use a symbolic link.
Add the directory in which `test_descartes` is to your PATH, if it is not already the case.

## Running the program

### Original program

Using the default polynomial (it displays the target curve in red):

```
python3 main.py 64
```

Using the Chebychev basis:

```
python3 main.py 64 -cheb
```

Using another polynomial:

```
python3 main.py 512 -poly ../polys/random.poly
```

### Program with Taylor approximation

Using the default polynomial (it displays the target curve in red):

```
python3 taylor.py 64
```

Using another polynomial:

```
python3 taylor.py 512 -poly ../polys/random.poly
```

## Help

For additional information use:

```
python3 main.py --help
```

or

```
python3 taylor.py --help
```