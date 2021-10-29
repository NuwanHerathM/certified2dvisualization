using Plots
using ImplicitEquations
import ArgParse
import DelimitedFiles

# Argument parsing
s = ArgParse.ArgParseSettings()
@ArgParse.add_arg_table s begin
    "N"
        help = "log2 of the resolution"
        arg_type = Int
        required = true
    "poly"
        help = "path to the polynomial file"
        required = true
end

parsed_args = ArgParse.parse_args(ARGS, s) # takes 0.5s no matter what

# Polynomial loading
mat = DelimitedFiles.readdlm(parsed_args["poly"])
(n,m) = size(mat)

g(x,y) = sum(x^i * sum(y^j * mat[i+1,j+1] for j in 0:n-1) for i in 0:m-1)

# Computation of the plot
p = plot(g ⩵ 0, xlim=(-1,1), ylim=(-1,1), N=parsed_args["N"], M=parsed_args["N"], aspect_ratio=:equal) 

# Saving
savefig(p,"image.png")
