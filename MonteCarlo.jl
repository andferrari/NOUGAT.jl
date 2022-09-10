using Plots, Distributions, LinearAlgebra
using NearestNeighbors 
using JLD2

include("my_funcs.jl")
pyplot()
# signal

include("params.jl")

# comp test statistics

# include("MonteCarlo_distrib.jl")

# jldsave("MonteCarlo.jld2"; t_nougat, t_rulsif, t_ma, t_knn)
MC = load("MonteCarlo.jld2")
t_nougat = MC["t_nougat"]
t_rulsif = MC["t_rulsif"]
t_ma = MC["t_ma"]
t_knn = MC["t_knn"]

# comp ROC

nc_detect = nc - n_ref - n_test
t_burn = 100 # à vérifier (utile que pour nougat) !

t0_nougat = t_nougat[t_burn:nc_detect-1,:]
t0_rulsif = t_rulsif[t_burn:nc_detect-1,:]
t0_ma = t_ma[t_burn:nc_detect-1,:]
t0_knn = t_knn[t_burn:nc_detect-1,:]

t1_nougat = t_nougat[nc_detect:end,:]
t1_ma = t_ma[nc_detect:end,:]
t1_rulsif = t_rulsif[nc_detect:end,:]
t1_knn = t_knn[nc_detect:end,:]

# comp PFA

pfa_nougat, ξ_nougat = comp_pfa(t0_nougat)
pfa_ma, ξ_ma = comp_pfa(t0_ma)
pfa_rulsif, ξ_rulsif = comp_pfa(t0_rulsif)
pfa_knn, ξ_knn = comp_pfa(t0_knn)

# comp MTFA

mtfa_nougat = comp_mtd(t0_nougat, ξ_pl=ξ_nougat)[1]
mtfa_ma = comp_mtd(t0_ma, ξ_pl=ξ_ma)[1]
mtfa_rulsif = comp_mtd(t0_rulsif, ξ_pl=ξ_rulsif)[1]
mtfa_knn = comp_mtd(t0_knn, ξ_pl=ξ_knn)[1]

# comp MTD

mtd_nougat = comp_mtd(t1_nougat, ξ_pl= ξ_nougat)[1] .+ (n_ref .+ n_test)
mtd_ma = comp_mtd(t1_ma, ξ_pl= ξ_ma)[1] .+ (n_ref .+ n_test)
mtd_rulsif = comp_mtd(t1_rulsif, ξ_pl= ξ_rulsif)[1] .+ (n_ref .+ n_test)
mtd_knn = comp_mtd(t1_knn, ξ_pl= ξ_knn)[1] .+ (n_ref .+ n_test)

# comp ROC
pfa_roc_nougat, pd_roc_nougat, toto = comp_roc(t0_nougat, t1_nougat; ξ_pl = ξ_nougat)
pfa_roc_rulsif, pd_roc_rulsif, toto = comp_roc(t0_rulsif, t1_rulsif; ξ_pl = ξ_rulsif)
pfa_roc_ma, pd_roc_ma, toto = comp_roc(t0_ma, t1_ma; ξ_pl = ξ_ma)
pfa_roc_knn, pd_roc_knn, toto = comp_roc(t0_knn, t1_knn; ξ_pl = ξ_knn)

# plots
Plots.reset_defaults()
Plots.scalefontsizes(1.1)

p1 = plot(n_ref + n_test + 1:nt, mean(t_nougat, dims=2), ribbon = std(t_nougat, dims=2), label = "NOUGAT", w=2, fillalpha=0.3, legend=:topleft)
plot!([nc], seriestype = :vline, label="", w=2)
plot!([nc+n_test], seriestype = :vline, label="", w=2)

p2 = plot(n_ref + n_test + 1:nt, mean(t_rulsif, dims=2), ribbon = std(t_rulsif, dims=2), label = "dRuLSIF", w=2, fillalpha=0.3, legend=:topleft)
plot!([nc], seriestype = :vline, label="", w=2)
plot!([nc+n_test], seriestype = :vline, label="", w=2)

p3 = plot(n_ref + n_test:nt, mean(t_ma, dims=2), ribbon = std(t_ma, dims=2), label = "MA", w=2, fillalpha=0.3, legend=:topleft)
plot!([nc], seriestype = :vline, label="", w=2)
plot!([nc+n_test], seriestype = :vline, label="", w=2)

p4 = plot(n_ref + n_test:nt, mean(t_knn, dims=2), ribbon = std(t_knn, dims=2), label = "k-NN", w=2, fillalpha=0.3, legend=:topleft)
plot!([nc], seriestype = :vline, label="", w=2)
plot!([nc+n_test], seriestype = :vline, label="", w=2)

plot(p1, p2, p3, p4, layout=(4, 1), show = true)
savefig("AllStatistics.pdf")

# MTFA
Plots.reset_defaults()
Plots.scalefontsizes(1.5)

plot(pfa_nougat, mtfa_nougat, label = "NOUGAT", w=2)
plot!(pfa_ma, mtfa_ma, label = "MA", w=2)
plot!(pfa_rulsif, mtfa_rulsif, label = "dRuLSIF", w=2)
plot!(pfa_knn, mtfa_knn, label = "k-NN", w=2, xlim=(0.02, 0.2), ylim=(30, 90))
xlabel!("PFA")
ylabel!("MTFA")
savefig("mtfa.pdf")



# MTFD
Plots.reset_defaults()
Plots.scalefontsizes(1.5)

plot(pfa_nougat, mtd_nougat, label = "NOUGAT", w=2)
plot!(pfa_ma, mtd_ma, label = "MA", w=2)
plot!(pfa_rulsif, mtd_rulsif, label = "dRuLSIF", w=2)
plot!(pfa_knn, mtd_knn, label = "k-NN", w=2,xlim=(0.01, 0.2)) #, ylim=:auto)
xlabel!("PFA")
ylabel!("MTD")
savefig("mtfd.pdf")

# ROC
pfa = [pfa_roc_nougat pfa_roc_rulsif pfa_roc_ma pfa_roc_knn]
pd = [pd_roc_nougat pd_roc_rulsif pd_roc_ma pd_roc_knn]
labels = ["NOUGAT" "dRuLSIF" "MA" "k-NN"]

Plots.reset_defaults()
Plots.scalefontsizes(1.5)

plot(pfa, pd, label=labels,  w=2, xlims=(0, 0.2), legend = :bottomright)
xlabel!("PFA")
ylabel!("PD")
lens!([0, 0.01], [0.9, 1], inset=(1, bbox(0.2, 0.2, 0.45, 0.5)), xtickfontsize=10, ytickfontsize=10)
#plot!(pfa, pd, xlims=(0, 0.01), ylims=(0.9, 1), w=2, inset=(1, bbox(0.2, 0.2, 0.45, 0.5)), xtickfontsize=10, ytickfontsize=10, label="", subplot=2)

savefig("roc.pdf")

