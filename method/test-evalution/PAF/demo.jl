using MAT, Graphs, SparseArrays, SimpleWeightedGraphs, LinearAlgebra
using DataFrames, CSV
using JLD, HDF5

include("genGraph_t1.jl")
include("BregmanOrig_v1.jl")
include("compute_violations.jl")
include("compute_nmse.jl")

function process_single_matrix(D0, N, output_prefix)
    """
    Processes a single distance matrix D0 using BregmanOrig_v1 and saves the results.

    Args:
        D0: The input distance matrix.
        N: The number of iterations.
        output_prefix: The prefix for output file names.
    """

    Dpaf, nmse, violation, update, sec = BregmanOrig_v1(copy(D0), N)

    # Output csv
    # route_csv = string(output_prefix, ".csv")
    # df = DataFrame(niter=0:N, nmse=nmse, violation=violation, update=update, sec=sec)
    # CSV.write(route_csv, df)
    # println("Saved CSV: ", route_csv)
    # 将每个矩阵保存为单独的 CSV 文件
    CSV.write("$(output_prefix)_nmse.csv", DataFrame(nmse, :auto))
    CSV.write("$(output_prefix)_violation.csv", DataFrame(violation, :auto))
    CSV.write("$(output_prefix)_update.csv", DataFrame(update, :auto))
    CSV.write("$(output_prefix)_sec.csv", DataFrame(sec, :auto))

    # Output JLD
    # route_jld = string(output_prefix, ".jld2") # Use jld2 for better compatibility
    # save(route_jld, "Dpaf", Dpaf, "nmse", nmse, "violation", violation, "update", update, "sec", sec)
    # println("Saved JLD2: ", route_jld)

    # # Output MAT (requires MAT.jl)
    # route_mat = string(output_prefix, ".mat")
    # matwrite(route_mat, Dict(
    #     "Dpaf" => Dpaf,
    #     "nmse" => nmse,
    #     "violation" => violation,
    #     "update" => update,
    #     "sec" => sec
    # ))
    # println("Saved MAT: ", route_mat)

end


# Example usage:
sample_size = 500

input_file = "data/graph$(sample_size).mat"
output_prefix = "result/$(sample_size)/"
N = 20 # Number of iterations
data = matread(input_file)
D0 = data["D"]
start_time = time()
process_single_matrix(D0, N, output_prefix)
end_time = time()
println("time ",end_time-start_time)
