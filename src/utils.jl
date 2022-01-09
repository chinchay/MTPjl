#%%
using JuLIP # for `atoms` in Julia https://github.com/JuliaMolSim/JuLIP.jl
using NeighbourLists # for `PairList()`
using LinearAlgebra # for `dot()`
using LoopVectorization # for @turbo

function getParam(lines::Vector{String}, iLine::Int)
    return strip( split(lines[iLine], "=")[2] , ' ')
end

function getParam2( lines::Vector{String},
                    n::Int,
                    iLine::Int,
                    typeNumber::String
    )
    if typeNumber == "Int"
        v = zeros(Int, n)
    elseif typeNumber == "Float64"
        v = zeros(Float64, n)
    end
    
    l = split( strip( getParam(lines, iLine), ['{', '}'] ), ',' )
    for i in 1:n
        if typeNumber == "Int"
            v[i] = parse(Int, l[i])
        elseif typeNumber == "Float64"
            v[i] = parse(Float64, l[i])
        end
    end
    return v
end    

function getParam3(lines::Vector{String}, n::Int, iLine::Int)
    v = zeros(Int, (n, 4))
    l = split( strip( getParam(lines, iLine), ['{', '}'] ), "}, {"  )
    for i in 1:n
            temp = split( l[i], ',' )
            for j in 1:4
                v[i, j] = parse(Int, temp[j])
            end
    end
    return v
end    

function load(filename="workdir/pot.mtp"::String)
    lines = readlines(filename)

    @assert lines[1] == "MTP" "Can read only MTP format potentials"

    version = getParam(lines, 2)
    @assert version == "1.1.0" "MTP file must have version \"1.1.0\""

    potential_name     = getParam(lines, 3)
    scaling            = parse(Float64, getParam(lines, 4))
    species_count      = parse(Int, getParam(lines, 5))
    potential_tag      = getParam(lines, 6)
    radial_basis_type  = getParam(lines, 7)
    min_dist           = parse(Float64, getParam(lines, 8))
    max_dist           = parse(Float64, getParam(lines, 9))
    radial_basis_size  = parse(Int, getParam(lines, 10))
    radial_funcs_count = parse(Int, getParam(lines, 11))
    
    iline = 13
    # regression_coeffs = zeros( (species_count, species_count, radial_funcs_count, radial_basis_size) )
    regression_coeffs = zeros( (radial_basis_size, radial_funcs_count, species_count, species_count) )
    for s1 in 1:species_count
        for s2 in 1:species_count
            iline += 1
            for i in 1:radial_funcs_count
                list_t = split( strip( lines[iline], ['\t', '{', '}'] ), ',' )
                iline += 1
                for j in 1:radial_basis_size
                    # regression_coeffs[s1, s2, i, j] = parse(Float64, list_t[j])
                    regression_coeffs[j, i, s2, s1] = parse(Float64, list_t[j])
                end
            end
        end
    end
    #
    
    alpha_moments_count = parse(Int, getParam(lines, iline))
    
    alpha_index_basic_count = parse(Int, getParam(lines, iline + 1))
    alpha_index_basic       = getParam3(lines, alpha_index_basic_count, iline + 2)
    
    radial_func_max = maximum([ alpha_index_basic[i, 1] for i in 1:alpha_index_basic_count ])
    @assert radial_func_max == (radial_funcs_count - 1) "Wrong number of radial functions specified"

    alpha_index_times_count = parse(Int, getParam(lines, iline + 3))
    alpha_index_times       = getParam3(lines, alpha_index_times_count, iline + 4)
    
    alpha_scalar_moments = parse(Int, getParam(lines, iline + 5))
    alpha_moment_mapping = getParam2(lines, alpha_scalar_moments, iline + 6, "Int")

    alpha_count = alpha_scalar_moments + 1
    
    species_coeffs       = getParam2(lines, species_count, iline + 7, "Float64")
    moment_coeffs        = getParam2(lines, alpha_scalar_moments, iline + 8, "Float64")
    #
    params = Dict(
            "scaling" => scaling,
            "species_count" => species_count,
            "rbasis_type" => radial_basis_type,
            "min_dist" => min_dist,
            "max_dist" => max_dist,
            "rb_size" => radial_basis_size,
            "radial_func_count" => radial_funcs_count,
            "regression_coeffs" => regression_coeffs,
            "alpha_moments_count" => alpha_moments_count,
            "alpha_index_basic_count" => alpha_index_basic_count,
            "alpha_index_basic" => alpha_index_basic,
            "alpha_index_times" => alpha_index_times,
            "alpha_scalar_moments" => alpha_scalar_moments,
            "alpha_count" => alpha_count,
            # "linear_coeffs":linear_coeffs,
            "species_coeffs" => species_coeffs,
            "moment_coeffs" => moment_coeffs,
            "alpha_index_times_count" => alpha_index_times_count,
            "alpha_moment_mapping" => alpha_moment_mapping,
            )
    #
    return params
end

function init_vecs(parameters::Dict{String, Any})
    rb_vals     = zeros(parameters["rb_size"])
    rb_ders     = zeros(parameters["rb_size"])
    moment_vals = zeros(parameters["alpha_moments_count"])
    basis_vals  = zeros(parameters["alpha_count"])
    site_energy_ders_wrt_moments_ = zeros(parameters["alpha_moments_count"])
    
    mult        = 2.0 / (parameters["max_dist"] - parameters["min_dist"])
    
    max_alpha_index_basic = maximum(parameters["alpha_index_basic"]) + 1
    inv_dist_powers_ = zeros(max_alpha_index_basic)
    coords_powers_   = zeros( (3, max_alpha_index_basic) )

    linear_mults = ones(parameters["alpha_scalar_moments"])
    max_linear = 1e-10 * ones(parameters["alpha_scalar_moments"])

    # # calculate moment_vals[i] for i only in the following list:
    # alpha_index_basic_count = parameters["alpha_index_basic_count"]
    # alpha_index_times_count = parameters["alpha_index_times_count"]
    # alpha_index_times = parameters["alpha_index_times"]
    # alpha_moment_mapping = parameters["alpha_moment_mapping"]
    # l1 = [ i for i in range(alpha_index_basic_count) if i in alpha_moment_mapping ]
    # l2 = [ alpha_index_times[i, 0] for i in range(alpha_index_times_count) if alpha_index_times[i, 0] < alpha_index_basic_count ]
    # l3 = [ alpha_index_times[i, 1] for i in range(alpha_index_times_count) if alpha_index_times[i, 1] < alpha_index_basic_count ]
    # l4 = [ alpha_index_times[i, 3] for i in range(alpha_index_times_count) if alpha_index_times[i, 3] < alpha_index_basic_count ]    
    # l1 = l1 + list( set(l2) - set(l1) )
    # l1 = l1 + list( set(l3) - set(l1) )
    # i_moment_vals = l1 + list( set(l4) - set(l1) )

    alpha_index_basic_count = parameters["alpha_index_basic_count"]
    alpha_index_basic = parameters["alpha_index_basic"]
    lmu       = zeros(Int16, alpha_index_basic_count)
    lAlphaSum = zeros(Int16, alpha_index_basic_count)
    for i in 1:alpha_index_basic_count
        lmu[i] = alpha_index_basic[i, 1]
        a1     = alpha_index_basic[i, 2]
        a2     = alpha_index_basic[i, 3]
        a3     = alpha_index_basic[i, 4]
        lAlphaSum[i] = a1 + a2 + a3
    end

    lval = zeros( parameters["alpha_index_basic_count"] )


    initializedVecs = Dict{String, Any}(
                "rb_vals" => rb_vals,
                "rb_ders" => rb_ders,
                "moment_vals" => moment_vals,
                "basis_vals" => basis_vals, 
                "site_energy_ders_wrt_moments_" => site_energy_ders_wrt_moments_, 
                "max_alpha_index_basic" => max_alpha_index_basic,
                "inv_dist_powers_" => inv_dist_powers_, 
                "coords_powers_" => coords_powers_,
                "linear_mults" => linear_mults,
                "max_linear" => max_linear,
                "mult" => mult,
                "lmu" => lmu,
                "lAlphaSum" => lAlphaSum,
                "lval" => lval

    )
    #
    return initializedVecs
end

function correctIndexesInParameters!(parameters::Dict{String, Any})
    # mutates `parameters`
    # correcting to index starting in 1 in Julia, instead of 0 as in C++ (MTP code)
    for i in 1:parameters["alpha_index_basic_count"]
        parameters["alpha_index_basic"][i, 1] += 1 # = mu
    end
    for i in 1:parameters["alpha_index_times_count"]
        parameters["alpha_index_times"][i, 1] += 1 # val0 = moment_vals[ alpha_index_times[i, 0] ] in finish_moment_vals()
        parameters["alpha_index_times"][i, 2] += 1 # val1 = moment_vals[ alpha_index_times[i, 1] ]
        parameters["alpha_index_times"][i, 4] += 1 # moment_vals[alpha_index_times[i, 3]] += val2 * val0 * val1
    end
    for i in 1:parameters["alpha_scalar_moments"]
        parameters["alpha_moment_mapping"][i] += 1
    end
end

function belongs(x::Float64, xmin::Float64, xmax::Float64)
    return (xmin <= x) && (x <= xmax)
end

# rb_Calc!(r, min_dist, max_dist, mult, rb_vals, rb_ders, scaling, rb_size)
function rb_Calc!(
                    r::Float64,
                    min_dist::Float64,
                    max_dist::Float64,
                    mult::Float64,
                    rb_vals::Vector{Float64},
                    rb_ders::Vector{Float64},
                    scaling::Float64,
                    rb_size::Int
    )
    # mutates rb_vals, rb_ders
    # from src/radial_basis.cpp: void RadialBasis_Chebyshev::RB_Calc(double r)
    
    # if parameters["rbasis_type"] == "RBChebyshev":
    @assert belongs(r, min_dist, max_dist) "r does not belong to 
                [min_dist, max_dist]: " * string(r) * " min_dist=" * 
                string(min_dist) * " max_dist=" * string(max_dist)
    #

    ksi = ( (2 * r) - (min_dist + max_dist)) / (max_dist - min_dist)
    R   = r - max_dist
    R2  = R ^ 2
    
    # scaling is not read in RadialBasis_Chebyshev::RB_Calc(), so it keeps its default value = 1
    rb_vals[1] = R2 # instead of `scaling * (1 * R2)``
    # rb_ders[0] = scaling * (0 * R2 + 2 * R) <<< why 0 * something??? I have to simplify below
    # rb_ders[1] = 2.0 * R

    rb_vals[2] = ksi * R2
    # rb_ders[2] = (mult * R2) + (2.0 * ksi * R)

    for i in 3:rb_size
        v1 = rb_vals[i - 1]
        v2 = rb_vals[i - 2]
        # d1 = rb_ders[i - 1]
        # d2 = rb_ders[i - 2]
        rb_vals[i] = (2.0 * ksi * v1) - v2
        # rb_ders[i] = 2.0 * ( (mult * v1) + (ksi * d1) ) - d2
    end
    # return rb_vals, rb_ders
end

# _pows(r, inv_dist_powers_, coords_powers_, max_alpha_index_basic, NeighbVect_j)
function _pows!(
    r::Float64,
    inv_dist_powers_::Vector{Float64},
    coords_powers_::Matrix{Float64},
    max_alpha_index_basic::Int64,
    x::Float64,
    y::Float64,
    z::Float64
    )
    # mutates inv_dist_powers_, coords_powers_
    inv_dist_powers_[1]  = 1.0
    coords_powers_[1, 1] = 1.0
    coords_powers_[2, 1] = 1.0
    coords_powers_[3, 1] = 1.0
    @inbounds for k in 2:max_alpha_index_basic
        inv_dist_powers_[k] = inv_dist_powers_[k - 1] / r
        coords_powers_[1, k] = coords_powers_[1, k - 1] * x
        coords_powers_[2, k] = coords_powers_[2, k - 1] * y
        coords_powers_[3, k] = coords_powers_[3, k - 1] * z
    end
    return inv_dist_powers_, coords_powers_
end

# Next: calculating non-elementary b_i
function finish_moment_vals!(
                moment_vals::Vector{Float64},
                alpha_index_times_count::Int,
                alpha_index_times::Matrix{Int64},
                )
    # mutates moment_vals
    @inbounds for i in 1:alpha_index_times_count ## ` @inbounds for ... `
        val0 = moment_vals[ alpha_index_times[i, 1] ] # float
        val1 = moment_vals[ alpha_index_times[i, 2] ] # float
        val2 = alpha_index_times[i, 3]                # integer
        m = val2 * val0 * val1
        moment_vals[alpha_index_times[i, 4]] += m  #*** <<<<<<<<<<<<<<<<<<<<< Optimize performance here!!!
    end
end

# https://discourse.julialang.org/t/memory-allocation-in-dot-product/67901
function mydot1(regression_coeffs, B, mu, type_outer, type_central, rb_size)
    # @assert length(A) == length(B)
    s = 0.0
    # @turbo for i ∈ eachindex(B)
    # @turbo for i in 1:rb_size
    @turbo for i in 1:rb_size
        @inbounds s += regression_coeffs[i, mu, type_outer, type_central] * B[i]
    end
    return s
end

function update_moment_vals!(
                    moment_vals::Vector{Float64},
                    alpha_index_basic_count::Int,
                    alpha_index_basic::Matrix{Int64},
                    inv_dist_powers_::Vector{Float64},
                    regression_coeffs::Array{Float64, 4},
                    type_central::Int8,
                    type_outer::Int8,
                    rb_vals::Vector{Float64},
                    rb_ders::Vector{Float64},
                    coords_powers_::Matrix{Float64},
                    rb_size::Int,
                    lmu::Vector{Int16},
                    lAlphaSum::Vector{Int16},
                    lval::Vector{Float64},
                    )
    # update_moment_vals
    #
    @inbounds for i in 1:alpha_index_basic_count
        mu = lmu[i]
        val = mydot1(regression_coeffs, rb_vals, mu, type_outer, type_central, rb_size)
        lval[i] = val
    end

    shiftIndx = 1 #0 # to account for Julia indexes begining in 1, instead of 0 as in C++
    @turbo for i in 1:alpha_index_basic_count
        a1 = alpha_index_basic[i, 2]
        a2 = alpha_index_basic[i, 3]
        a3 = alpha_index_basic[i, 4]
        
        pow0 = coords_powers_[1, a1 + shiftIndx]
        pow1 = coords_powers_[2, a2 + shiftIndx]
        pow2 = coords_powers_[3, a3 + shiftIndx]
        mult0 = pow0 * pow1 * pow2

        # k = a1 + a2 + a3
        k = lAlphaSum[i]
        inv_powk = inv_dist_powers_[k + shiftIndx] # I had to +1 to account for indices beginning in 1, not zero.
        val = lval[i]
        val *= inv_powk
        # der = (der * inv_powk) - (k * val / r)

        moment_vals[i] += val * mult0  ## ***
        # mult0 *= der / r
        # moment_jacobian_[i, j, 0] += mult0 * NeighbVect_j[0]
        # moment_jacobian_[i, j, 1] += mult0 * NeighbVect_j[1]
        # moment_jacobian_[i, j, 2] += mult0 * NeighbVect_j[2]
        
        # if a1 != 0:
        #     prod = val * a1 * coords_powers_[a1 - 1, 0] * pow1 * pow2
        #     moment_jacobian_[i, j, 0] += prod
        # #
        # if a2 != 0:
        #     prod = val * a2 * pow0 * coords_powers_[a2 - 1, 1] * pow2
        #     moment_jacobian_[i, j, 1] += prod
        # #
        # if a3 != 0:
        #     prod = val * a3 * pow0 * pow1 * coords_powers_[a3 - 1, 2]
        #     moment_jacobian_[i, j, 2] += prod
        # #
    end
end

function calcSiteEnergyDers(
    iAtom::Int,
    numberOfNeighbors::Vector{Int16},
    l_xyzr::Array{Float64, 3},
    l_t::Matrix{Int8},
    type_central::Int8, # type of central atom at `i`
    species_coeffs::Vector{Float64},
    moment_coeffs::Vector{Float64},
    alpha_index_basic::Matrix{Int64},
    alpha_index_basic_count, 
    max_alpha_index_basic,
    alpha_index_times::Matrix{Int64},
    alpha_index_times_count::Int,
    alpha_scalar_moments::Int,
    alpha_moment_mapping::Vector{Int64},
    regression_coeffs::Array{Float64, 4},
    alpha_moments_count::Int,
    moment_vals::Vector{Float64},
    inv_dist_powers_::Vector{Float64},
    coords_powers_::Matrix{Float64},
    species_count::Int,
    radial_func_count::Int,
    rb_size::Int,
    min_dist::Float64,
    max_dist::Float64,
    mult::Float64,
    rb_vals::Vector{Float64},
    rb_ders::Vector{Float64},
    scaling::Float64,
    linear_mults::Vector{Float64},
    max_linear::Vector{Float64},
    lmu::Vector{Int16},
    lAlphaSum::Vector{Int16},
    lval::Vector{Float64},
    )
    #
    # from dev_src/mtpr.cpp: void MLMTPR::CalcSiteEnergyDers(const Neighborhood& nbh)
    #

    # initialize
    buff_site_energy_ = 0.0
    # moment_vals .= 0.0


    # lenNbh = len(nbh)

    # dicTypes = {"C":0, "O": 1}
    # types = [0,0,0,0, 1,1] #just an example

    # C = species_count         #number of different species in current potential
    # K = radial_func_count     #number of radial functions in current potential
    # R = rb_size                 #number of Chebyshev polynomials constituting one radial function

    # moment_jacobian_ = np.zeros((alpha_index_basic_count, lenNbh, 3))

    @assert type_central < species_count "Too few species count in the MTP potential!"

    neighs_i = numberOfNeighbors[iAtom]
    @inbounds for j in 1:neighs_i
        x = l_xyzr[1, j, iAtom]
        y = l_xyzr[2, j, iAtom]
        z = l_xyzr[3, j, iAtom]
        r = l_xyzr[4, j, iAtom]
        type_outer = l_t[j]

        # mutates rb_vals, rb_ders
        rb_Calc!(r, min_dist, max_dist, mult, rb_vals, rb_ders, scaling, rb_size)

        rb_vals .*= scaling # rb_vals is numpy array, so we can directly multyply by a float if array is float
        # rb_ders .*= scaling

        # mutates inv_dist_powers_, coords_powers_
        _pows!(r, inv_dist_powers_, coords_powers_, max_alpha_index_basic, x, y, z) ## ***

        # mutates moment_vals  ## ***
        update_moment_vals!(
                            moment_vals,
                            alpha_index_basic_count,
                            alpha_index_basic,
                            inv_dist_powers_,
                            regression_coeffs,
                            type_central,
                            type_outer,
                            rb_vals,
                            rb_ders,
                            coords_powers_,
                            rb_size,
                            lmu,
                            lAlphaSum,
                            lval
                            )
        #

        ## Repulsive term
        ## I think it was not implemented in the C++ MTP code)
        ## if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls")
        ## this seems arbitrary, I removed it:
        ## if r < min_dist:
        ##  multiplier = 10000;
        ##  buff_site_energy_ += multiplier*(exp(-10*(r-1)) - exp(-10*(min_dist-1)))
        ##  for (int a = 0; a < 3; a++)
        ##      buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (r - 1))/ nbh.dists[j])*nbh.vecs[j][a];
        # #
    end

    # # Next: calculating non-elementary b_i
    # for i in range(alpha_index_times_count):
    #     val0 = moment_vals[ alpha_index_times[i, 0] ] # float
    #     val1 = moment_vals[ alpha_index_times[i, 1] ] # float
    #     val2 = alpha_index_times[i, 2]                # integer
    #     moment_vals[alpha_index_times[i, 3]] += val2 * val0 * val1
    # #
    # mutates moment_vals
    finish_moment_vals!(moment_vals, alpha_index_times_count, alpha_index_times)

    # renewing maximum absolute values
    # for i in range(alpha_scalar_moments):
        # max_linear[i] = max(
        #     max_linear[i],
        #     abs(linear_coeffs[species_count + i] * moment_vals[alpha_moment_mapping[i]])
        #     )
        # #
    #

    # convolving with coefficients
    buff_site_energy_ += species_coeffs[type_central]

    @inbounds for i in 1:alpha_scalar_moments
        # buff_site_energy_ += linear_coeffs[species_count + i] * linear_mults[i] * moment_vals[alpha_moment_mapping[i]]
        # I simplified because linear_mults[i] = 1
        buff_site_energy_ += moment_coeffs[i] * moment_vals[alpha_moment_mapping[i]]
    end

    return buff_site_energy_
end

function CalcEFS(
    nAtoms::Int,
    numberOfNeighbors::Vector{Int16},
    l_xyzr::Array{Float64, 3},
    l_t::Matrix{Int8},
    l_t_centrals::Vector{Int8},
    params::Dict{String, Any},
    vecs::Dict{String, Any},
    )
    # from src/basic_mlip.cpp:  void AnyLocalMLIP::CalcEFS(Configuration& cfg)
    energy = 0.0

    species_coeffs = params["species_coeffs"]
    moment_coeffs = params["moment_coeffs"]
    alpha_index_basic = params["alpha_index_basic"]
    alpha_index_basic_count = params["alpha_index_basic_count"]
    max_alpha_index_basic = vecs["max_alpha_index_basic"]
    alpha_index_times = params["alpha_index_times"]
    alpha_index_times_count = params["alpha_index_times_count"]
    alpha_scalar_moments = params["alpha_scalar_moments"]
    alpha_moment_mapping = params["alpha_moment_mapping"]
    regression_coeffs = params["regression_coeffs"]
    inv_dist_powers_ = vecs["inv_dist_powers_"]
    coords_powers_ = vecs["coords_powers_"]
    species_count = params["species_count"]
    radial_func_count = params["radial_func_count"]
    rb_size = params["rb_size"]
    min_dist = params["min_dist"]
    max_dist = params["max_dist"]
    mult = vecs["mult"]
    rb_vals = vecs["rb_vals"]
    rb_ders = vecs["rb_ders"]
    scaling = params["scaling"]
    linear_mults = vecs["linear_mults"]
    max_linear = vecs["max_linear"]
    alpha_moments_count = params["alpha_moments_count"]
    moment_vals = vecs["moment_vals"]
    lmu = vecs["lmu"]
    lAlphaSum = vecs["lAlphaSum"]
    lval = vecs["lval"]

    for i in 1:nAtoms
        type_central = l_t_centrals[i]    
        moment_vals .= 0.0
        
        # calcSiteEnergyDers
        energy += calcSiteEnergyDers(
                    i,
                    numberOfNeighbors,
                    l_xyzr,
                    l_t,
                    type_central,
                    species_coeffs,
                    moment_coeffs,
                    alpha_index_basic,
                    alpha_index_basic_count,
                    max_alpha_index_basic,
                    alpha_index_times,
                    alpha_index_times_count,
                    alpha_scalar_moments,
                    alpha_moment_mapping,
                    regression_coeffs,
                    alpha_moments_count,
                    moment_vals,
                    inv_dist_powers_,
                    coords_powers_,
                    species_count,
                    radial_func_count,
                    rb_size,
                    min_dist,
                    max_dist,
                    mult,
                    rb_vals,
                    rb_ders,
                    scaling,
                    linear_mults,
                    max_linear,
                    lmu,
                    lAlphaSum,
                    lval
                )
        #
    #
    #
    end
    

    return energy
end

function my_neighbors(max_dist, atoms, nAtoms)
    # ni  = zeros(Int, nAtoms)
    C = zeros(Int, (nAtoms, nAtoms))
    for i in 1:nAtoms
        Ri = atoms.X[i]
        for j in (i+1):nAtoms
            Rji = atoms.X[j] - Ri
            if belongs( Rji[1], -max_dist, max_dist )
                if belongs( Rji[2], -max_dist, max_dist )
                    if belongs( Rji[3], -max_dist, max_dist )
                        C[i,j] = 1
                    end
                end
            end
        end
    end

        

end

function get_xyzrNeighs(
    nAtoms::Int,
    atoms::Atoms,
    cutoff::Float64,
    dictionaryTypes::Dict{Int8, Int8}
    )
    # 
    numberOfNeighbors = zeros(Int16, nAtoms)
    nlist = PairList(atoms.X, cutoff, atoms.cell, (true, true, true) )
    for i in 1:nAtoms
        list_j, list_D = neigs(nlist, i)
        n = length(list_j)
        numberOfNeighbors[i] = n

    end


#        # allocate arrays for the many threads
#    # set number of threads
#    nt, nn = setup_mt(nat)
#    # allocate arrays
#    first_t = Vector{TI}[ Vector{TI}()  for n = 1:nt ]    # i
#    secnd_t = Vector{TI}[ Vector{TI}()  for n = 1:nt ]    # j
#    shift_t = Vector{SVec{TI}}[ Vector{SVec{TI}}()  for n = 1:nt ]  # ~ X[i] - X[j]

end

function get_neighborhoods(
                        atoms::Atoms,
                        cutoff::Float64,
                        dictionaryTypes::Dict{Int8, Int8}
    )
    #
    nAtoms = length(atoms.X)
    nlist = PairList(atoms.X, cutoff, atoms.cell, (true, true, true) )
    nbs = []
    numberOfNeighbors = zeros(Int16, nAtoms)
    
    for i in 1:nAtoms
        list_j, list_D = neigs(nlist, i)
        n      = length(list_j)
        numberOfNeighbors[i] = n
        list_d = zeros(Float64, n)
        list_z = zeros(Int16, n) # atomic numbers
        for j in 1:n
            v = list_D[j]
            list_d[j] = sqrt( dot(v, v) )
            list_z[j] = dictionaryTypes[ atoms.Z[j] ]
        end
        append!( nbs, [( list_j,  list_d, list_D, list_z )] )
    end

    maxNeighs = maximum(numberOfNeighbors)
    l_xyzr    = zeros(Float64, (4, maxNeighs, nAtoms) )
    l_types   = zeros(Int8, (maxNeighs, nAtoms) )
    for i in 1:nAtoms
        for j in 1:numberOfNeighbors[i]
            l_xyzr[1, j, i] = nbs[i][3][j][1]
            l_xyzr[2, j, i] = nbs[i][3][j][2]
            l_xyzr[3, j, i] = nbs[i][3][j][3]
            l_xyzr[4, j, i] = nbs[i][2][j]
            l_types[j, i]    = nbs[i][4][j]
        end
    end
    # return nbs, numberOfNeighbors
    return numberOfNeighbors, l_xyzr, l_types
end

# type_centrals = get_type_centrals(atoms, dictionaryTypes)
function get_type_centrals(
                            atoms::Atoms,
                            dictionaryTypes::Dict{Int8, Int8}
    )
    # dictionaryTypes = { "C":0, "O":1 }
    type_centrals = zeros(Int8, length(atoms.X)) # Int16 as stated in line=21 in https://github.com/JuliaMolSim/JuLIP.jl/blob/master/src/chemistry.jl
    type_centrals = [dictionaryTypes[z] for z in atoms.Z]
    return type_centrals
end

#TODO: define nAtoms instead of using lenth(atoms)
#TODO: use `type_centrals` as argument in get_neighborhoods()
#TODO: lenNbh = nbh[1], delete this line, and arrange the arguments of calcSiteEnergyDers(..., lengNbh,...)
#TODO: sqrt() takes time, is it possible to get rid of it? knowing that 1/d will be needed later in _pows() ?