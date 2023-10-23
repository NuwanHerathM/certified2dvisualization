# Benchmarking

## Julia

For Julia, the sysimage `sys_plots.so` was created using the example at https://julialang.github.io/PackageCompiler.jl/dev/examples/plots.

Unfortunately, I am facing various problems when I try to create it again. So, I cannot give more information about that...

## Comparison to state-of-the-art implementations

The file `timing.sh` runs the benchmark, it stores the table as .tsv files in `output/` and the graphs as .png files in `images/`.\
For a given polynomial stored in `../polys/polynomial.poly`, the output is written in `output/polynomial.tsv`.\
For each polynomial, it consists in a double entry table with times for each method and each resolution. It has the following shape.

`output/polynomial.poly`:
| N      | resol1 | resol2 |
| :--- | ---: | ---: |
| methodA | timeA1 | timeA2 |
| methodB | timeB1 | timeB2 |

The script was writing for personal usage, so there is no guarantee that it might work for you.\
Nonetheless, the different files starting with `compare_` are here so that you can see what was tested and test it yourself.


## Test of the relevance of the IDCT for a fast multipoint evaluation

The file `partial_eval_timing.sh` compares the partial evaluation using error tracking with Horner's method and with the IDCT, from the two scripts `partial_eval_horner.py` and `partial_eval_dct.py`. A table is generated for each one.\
The outputs are stored in `output/partial_eval_horner.tsv` and `partial_eval_dct.tsv` as tables with the following shape.

`output/partial_eval_method.tsv`:
|  | polyA | polyB |
| :--- | ---: | ---: |
| resol1 | timeA1 | timeB1 |
| resol2 | timeA2 | timeB2 |
