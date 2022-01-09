module Calc

using JuLIP # for `atoms` in Julia https://github.com/JuliaMolSim/JuLIP.jl
using BenchmarkTools
include("utils.jl")

export MTP, getInfoNeighs, getEnergy

struct MTP <: AbstractCalculator
    params::Dict{String, Any}
    max_dist::Float64
    vecs::Dict{String, Any}
    dictionaryTypes::Dict{Int8, Int8}
end
@pot MTP
cutoff(mtp::MTP) = mtp.max_dist

function MTP( potfilename::String,  dictionaryTypes::Dict{Int8, Int8} ) # absolute path
    params = load(potfilename);
    max_dist = params["max_dist"];
    correctIndexesInParameters!(params);
    vecs = init_vecs(params);
    return MTP(params, max_dist, vecs, dictionaryTypes)
end

function getInfoNeighs(mtp::MTP, at::Atoms)
    nAtoms = length(at.X)
    l_t_centrals = get_type_centrals(at, mtp.dictionaryTypes);
    numberOfNeighbors, l_xyzr, l_types = get_neighborhoods(at, mtp.max_dist, mtp.dictionaryTypes);
    infoNeighs = [ nAtoms, numberOfNeighbors, l_xyzr, l_types, l_t_centrals ]
    return infoNeighs
end

function getEnergy(mtp::MTP, infoNeighs::Vector{Any})
    nAtoms, numberOfNeighbors, l_xyzr, l_types, l_t_centrals = infoNeighs
    E = CalcEFS(
        nAtoms,
        numberOfNeighbors,
        l_xyzr,
        l_types,
        l_t_centrals,
        mtp.params,
        mtp.vecs
        )
    # 
    return E
end

function getEnergy(mtp::MTP, at::Atoms)
    nAtoms, numberOfNeighbors, l_xyzr, l_types, l_t_centrals = getInfoNeighs(mtp, at)
    E = CalcEFS(
        nAtoms,
        numberOfNeighbors,
        l_xyzr,
        l_types,
        l_t_centrals,
        mtp.params,
        mtp.vecs
        )
    # 
    return E
end

#%% ****************************************
mutable struct MyCalc
    # MyCalc() = new() # constructor
    params::Dict{String, Any}
    max_dist::Float64
    vecs::Dict{String, Any}
    # atoms::Atoms
    # dictionaryTypes::Dict{Int8, Int8}
    # l_t_centrals::Vector{Int8}
    # numberOfNeighbors::Vector{Int16}
    # l_xyzr::Array{Float64, 3}
    # l_types::Matrix{Int8}
    # nAtoms::Int
    function MyCalc(potfilename::String) # absolute path
        params = load(potfilename);
        max_dist = params["max_dist"];
        correctIndexesInParameters!(params);
        vecs = init_vecs(params);
        new(params, max_dist, vecs)
    end
end

function loadAtoms(self::MyCalc, atoms::Atoms, dictionaryTypes::Dict{Int8, Int8})
    self.atoms = atoms
    self.dictionaryTypes = dictionaryTypes
    self.l_t_centrals = get_type_centrals(atoms, dictionaryTypes);
    self.numberOfNeighbors, self.l_xyzr, self.l_types = get_neighborhoods(self.atoms, self.max_dist, self.dictionaryTypes);
    self.nAtoms = length(atoms.X)

end

function getEnergy(self::MyCalc)
    energy = CalcEFS(
        self.nAtoms,
        self.numberOfNeighbors,
        self.l_xyzr,
        self.l_types,
        self.l_t_centrals,
        self.params,
        self.vecs
        )
    # 
    return energy
end




end