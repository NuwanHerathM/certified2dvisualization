# Title

This isn't an actual readme...
These are rather notes, in order to remind myself the main ways to run the program.

## Running the program

Using the default polynomial (it displays the target curve in red):

```
python3 subdivision.py 64
```

Using the Chebychev basis:

```
python3 subdivision.py 64 -cheb
```

Using another polynomial:

```
python3 subdivision.py 512 -poly ../polys/random.poly
```

## Profiling

### Prerequisite

https://github.com/rkern/line_profiler

### The main command required

```
kernprof -lv subdivision.py 1024 -poly ../polys/random.poly
```
