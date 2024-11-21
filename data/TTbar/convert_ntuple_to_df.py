#!/usr/bin/env python
#import numpy as np
# %% 
import pandas as pd

import uproot
import awkward as ak
import vector
vector.register_awkward()

from argparse import ArgumentParser

def get_jetconst_columns(jet_collection, n_const = 8):
    ''' Get constituent column names '''

    const_members = [
        'pt','eta','phi',
    #     'mass', # valid?
        'pdgId', # reco Id by PF algo, not full pdgID, but reduced variety for hadrons - has inefficiency! and not to be used directly
        'vz'
    ]

    const_columns = []

    for idau in range(n_const):
        const_columns += [jet_collection + "_dau%i_%s" %(idau,memb) for memb in const_members]

    return const_columns

def load_arrays(fname = "data/perfNano_TTbar_PU200_GenParts_More.root",
                jet_collection = "L1PuppiAK4Jets",
                n_const = 8,
                ):
    ''' Load arrays from ROOT file and return '''
    print("Reading arrays from %s" %fname)

    # open ROOT file
    tree = uproot.open(fname)["Events"]

    # get available jet collection names
    avail_jet_collections = [b[:-3] for b in tree.keys(filter_name="*Jets_pt")]

    ## check jet collections are present
    for colname in ["GenJets",jet_collection]:
        if colname not in avail_jet_collections:
            print(colname, " not available in ", fname)
            return 0

    ## get GenPart and GenJets
    gen_arrays = tree.arrays(library="ak", how="zip",
                             filter_name = "/(GenCands_|GenJets_)(pt|eta|phi|pdgId|status|partonFlavour|hadronFlavour)/"
                             )

    genparts = ak.with_name(gen_arrays.GenCands, "Momentum4D")
    genjets = ak.with_name(gen_arrays.GenJets, "Momentum4D")
    print("Loaded: %i events" %len(genjets))
    print("Loaded: %i genjets" %len(ak.flatten(genjets, axis = 1)))
    print("Loaded: %i genparts" %len(ak.flatten(genparts, axis = 1)))
    # get jets array for collection of interest
    if jet_collection == "GenJets":
        # no need to load anything extra, use genjets as reco jets
        l1jets = genjets
    else:
        jet_arrays = tree.arrays(library="ak", how="zip", filter_name = "/(%s_)(pt|eta|phi)/" %jet_collection )
        l1jets = ak.with_name(jet_arrays[jet_collection], "Momentum4D")
        print("Loaded: %i %s" %(len(ak.flatten(l1jets, axis = 1)), jet_collection))

    ## Load jet constituents
    const_columns = get_jetconst_columns(jet_collection, n_const)
    const_array = tree.arrays(library="ak", how="zip", filter_name = const_columns)
    jet_consts = const_array[jet_collection]

    return genparts, genjets, l1jets, jet_consts

def match_genParts(l1jets, genparts):
    ## The mapping of GenParts to be matched to
    gen_sel_ids = {
        ## light
        "g": (abs(genparts.pdgId) == 21) & (abs(genparts.status) == 71),
        "q": (abs(genparts.pdgId) < 5) & (abs(genparts.status) == 23),
        "b": (abs(genparts.pdgId) == 5) & (abs(genparts.status) == 23),
        ## heavy
        "t": (abs(genparts.pdgId) == 6) & (abs(genparts.status) == 22),
        "W": (abs(genparts.pdgId) == 24) & (abs(genparts.status) == 22),
        ## leptons
        "e": (abs(genparts.pdgId) == 11) & (abs(genparts.status) == 1),
        "mu": (abs(genparts.pdgId) == 13) & (abs(genparts.status) == 1),
        "tau": (abs(genparts.pdgId) == 15) & (abs(genparts.status) == 2),
    }


    ## matching to Gen Particles
    for part_label, part_sel in gen_sel_ids.items():

        jet_gen = ak.cartesian({"jets": l1jets, "gen": genparts[part_sel]}, nested=True)
        js, gs = ak.unzip(jet_gen)
        dR = js.deltaR(gs)

        l1jets["min_dR_" + part_label] = ak.min(dR, axis = -1)
        print(part_label, ak.sum(l1jets["min_dR_" + part_label] < 0.4))

    return l1jets

def match_genJets(l1jets, genjets):

    jet_gen = ak.cartesian({"jets": l1jets, "gen": genjets}, nested=True)
    js, gs = ak.unzip(jet_gen)
    dR = js.deltaR(gs)

    l1jets["min_dR_genjet"] = ak.min(dR, axis = -1)
    best_dR = ak.argmin(dR, axis=-1, keepdims=True)

    l1jets["partonFlavour"] = jet_gen[best_dR].gen.partonFlavour
    l1jets["hadronFlavour"] = jet_gen[best_dR].gen.hadronFlavour
    print (l1jets["hadronFlavour"])
    return l1jets

# %%
def convert(
        fname = "data/perfNano_TTbar_PU200_GenParts_More.root",
        jet_collection = "L1PuppiAK4Jets",
        n_const = 8,
        outfname = None,):

    print("## Loading arrays")
    print (fname,jet_collection,n_const)
    genparts, genjets, l1jets, jet_consts = load_arrays(fname,jet_collection,n_const)

    print("## Matching genparts")
    ## match to GenParticles
    l1jets = match_genParts(l1jets, genparts)

    ## match to GenJets
    if jet_collection != "GenJets":
        print("## Matching GenJets")
        l1jets = match_genJets(l1jets, genjets)

    ## save match info separately from data (to avoid using it in training!)
    vec_fields = ["pt","eta","phi"]
    match_fields = [f for f in l1jets.fields if not (f in vec_fields)]

    print("Making dataframes")
    ## make dataframes (takes longest)
    # matching info separately
    df_match = ak.to_pandas(l1jets[match_fields])

    # global jet vars
    df_jets = ak.to_pandas(l1jets[vec_fields])
    # constituents
    df_consts = ak.to_pandas(jet_consts)

    # join jet vars and constituents
    df_jets = df_jets.join(df_consts)

    # drop spurious subsubentry from index
    #df_match.index = df_match.index.droplevel(2)
    #df_jets.index = df_jets.index.droplevel(2)

    ## Write out
    if outfname is None:
        outfname = fname.replace(".root","_%s_nConst%i.h5" % (jet_collection,n_const))
    print("Writing dataframes to %s" %outfname)

    # save the match and jet data separately
    df_match.to_hdf(outfname, key = "geninfo")
    df_jets.to_hdf(outfname, key = "data")

    return 1

if __name__ == "__main__":

    parser = ArgumentParser()
    # parser arguments
    parser.add_argument("infname", type=str, help="Input file name")
    parser.add_argument("-o", "--outfname", type=str, default=None, help="Output file name")
    parser.add_argument("-c", "--collection", type=str, default="L1PuppiAK4Jets", help="Jet Collection to be used", choices=['GenJets','L1PuppiAK4Jets', 'scPuppiCorrJets', 'L1PuppiSC4Jets'])
    parser.add_argument("-n", "--n_const", type=int, default=8, help="Number of constituents to read")

    args = parser.parse_args()
    convert(args.infname, args.collection, args.n_const, args.outfname)


# %%
