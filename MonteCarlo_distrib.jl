using SharedArrays

@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using NearestNeighbors
@everywhere include("my_funcs.jl")

include("params.jl")
# comp test statistics

realmax = 1000000

t_nougat = SharedArray{Float64,2}((nt - n_ref - n_test, realmax))
t_rulsif = SharedArray{Float64,2}((nt - n_ref - n_test, realmax))
t_ma =  SharedArray{Float64,2}((nt - n_ref - n_test + 1, realmax))
t_knn =  SharedArray{Float64,2}((nt - n_ref - n_test + 1, realmax))

@sync @distributed for k in 1:realmax
    (k % 100 == 0) && println("> ", k)

    dict = [rand(pdf_h0, 40) rand(pdf_h1, 40)]

    x = hcat(rand(pdf_h0, nc - 1), rand(pdf_h1, nt - nc + 1))

    t_nougat[:, k] = nougat(x, dict, n_ref, n_test, μ, ν, γ)
    t_rulsif[:, k] = rulsif(x, dict, n_ref, n_test, ν, γ)
    t_ma[:, k] = ma(x, dict, n_ref, n_test, γ)
    t_knn[:, k] = knnt(x, n_ref, n_test, k_knn)  
end
