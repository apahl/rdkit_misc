# scaf_frag.py
"""
Generating fragments out of molecules and scaffolds.

Axel Pahl, 10-Oct-2018
"""
# from collections import Counter

import pickle

import pandas as pd

from rdkit.Chem import AllChem as Chem
import rdkit.Chem.Descriptors as Desc
import rdkit.Chem.Scaffolds.MurckoScaffold as MurckoScaffold
from rdkit.Chem import Draw
Draw.DrawingOptions.atomLabelFontFace = "DejaVu Sans"
Draw.DrawingOptions.atomLabelFontSize = 18


class Fragments():
    def __init__(self):
        self.data = {}

    def add_fragment(self, mol_or_smi, is_active: bool, min_ha=0, max_ha=50, cpd_id=None):
        """e.g.: min_ha=8, max_ha=26 (inclusive)"""
        if isinstance(mol_or_smi, str):
            mol = Chem.MolFromSmiles(mol_or_smi)
            smi = mol_or_smi
        else:
            mol = mol_or_smi
            smi = Chem.MolToSmiles(mol_or_smi)
        if not mol:
            print("No valid Mol object.")
            return
        if smi in self.data:
            self.data[smi]["Count"] += 1
            if is_active:
                self.data[smi]["Active"] += 1
            if cpd_id is not None:
                self.data[smi]["Present_In"] = self.data[smi]["Present_In"] + " " + str(cpd_id)
        else:
            ha_count = Desc.HeavyAtomCount(mol)
            if ha_count < min_ha or ha_count > max_ha:
                return
            self.data[smi] = {}
            self.data[smi]["HA"] = ha_count
            self.data[smi]["Rings"] = Desc.RingCount(mol)
            self.data[smi]["Count"] = 1
            self.data[smi]["Active"] = 0
            if is_active:
                self.data[smi]["Active"] += 1
            if cpd_id is not None:
                self.data[smi]["Present_In"] = str(cpd_id)

    def add_fragments(self, mols, is_active: bool, min_ha=0, max_ha=50, cpd_id=None):
        for mol in mols:
            self.add_fragment(mol, is_active, min_ha, max_ha, cpd_id=cpd_id)

    def calc_perc_act(self):
        """Calculate the relative activity ratio in percent "PercAct" per fragment.
        Operates in-place!"""
        for smi in self.data:
            self.data[smi]["PercAct"] = 100 * self.data[smi]["Active"] / self.data[smi]["Count"]

    def filter(self, prop, cutoff, less_or_equal=True):
        """When less_or_equal == True, frags with prop <= cutoff are returned,
        otherwise frags with prop > cutoff are returned.
        Pre-defined props are: "HA", "Rings", "Count", "Active" & "PercAct" (after calc_perc_act() was run)."""
        result = Fragments()
        for smi in self.data:
            if less_or_equal:
                if self.data[smi][prop] <= cutoff:
                    result.data[smi][prop] = self.data[smi][prop]
            else:
                if self.data[smi][prop] > cutoff:
                    result.data[smi][prop] = self.data[smi][prop]
        return result

    def copy(self, props: list):
        """Returns a partial copy of the Fragments with just the given props."""
        result = Fragments()
        for smi in self.data:
            result.data[smi] = {}
            for prop in props:
                if prop in self.data[smi]:
                    result.data[smi][prop] = self.data[smi][prop]
        return result


    def save(self, fn: str):
        """Save Fragment object as pickle.
        The file ending `.pkl` is automatically added, if not present."""
        if not fn.endswith(".pkl"):
            fn = fn + ".pkl"
        with open(fn, "wb") as f:
            pickle.dump(self.data, f)


    def to_df(self) -> pd.DataFrame:
        result = pd.DataFrame.from_dict(self.data, orient="index")
        result = result.reset_index()
        result = result.rename(columns={"index": "Smiles"})
        return result


def load_fragments(fn: str) -> Fragments:
    """Load Fragment object from pickle.
    The file ending `.pkl` is automatically added, if not present."""
    if not fn.endswith(".pkl"):
            fn = fn + ".pkl"
    result = Fragments()
    with open(fn, "rb") as f:
        result.data = pickle.load(f)
    return result


def get_exhaustive_frags(mol):
    """Exhaustively split a molecule into fragments.
    Only-non-ring bonds are split and only fragments with at least one ring
    are considered to be a fragment.
    Returns a list of fragments as Mols."""
    result = []
    smiles = []
    bonds_to_break = []
    for b in mol.GetBonds():
        if not b.IsInRing():
            bonds_to_break.append(b.GetIdx())
    nm = Chem.FragmentOnBonds(mol, bonds_to_break)
    frags = Chem.GetMolFrags(nm, asMols=True)
    for f in frags:
        if Chem.CalcNumRings(f) > 0:
            murcko = MurckoScaffold.GetScaffoldForMol(f)
            smi = Chem.MolToSmiles(murcko)
            if smi not in smiles:
                smiles.append(smi)
                result.append(murcko)
    return result


def get_hieriarchical_frags(mol_or_smi):
    """Hierarchically (recursively) split a molecule into fragments.
    Only-non-ring bonds are split and only fragments with at least one ring
    are considered.
    Takes a mol object or a Smiles string as input.
    Returns a list of fragments as Smiles."""

    def _recursive_split(s, n=0):
        m = Chem.MolFromSmiles(s)
        if m is None: return
        splittable_bonds = []
        for b in m.GetBonds():
            if not b.IsInRing():
                splittable_bonds.append(b.GetIdx())
        frags = []
        for bidx in splittable_bonds:
            nm = Chem.FragmentOnBonds(m, [bidx], addDummies=False)
            try:
                splits = Chem.GetMolFrags(nm, asMols=True)
            except ValueError:
                continue
            # verify the split occurred between two rings
            if len(splits) == 2 and Chem.CalcNumRings(splits[0]) > 0 and Chem.CalcNumRings(splits[1]) > 0:
                frags.extend(splits)
        for f in frags:
            try:
                murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=f)
            except ValueError:
                continue
            if murcko not in result:
                result[murcko] = True
                if "[CH]" in murcko:
                    print(f"{murcko}  ({Chem.MolToSmiles(f)})")
                _recursive_split(murcko, n + 1)

    if isinstance(mol_or_smi, str):
        try:
            murcko = MurckoScaffold.MurckoScaffoldSmiles(smiles=mol_or_smi)
        except ValueError:
            return []
    else:
        try:
            murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol_or_smi)
        except ValueError:
            return []
    result = {murcko: True}
    _recursive_split(murcko)
    return list(sorted(result.keys(), key=len, reverse=True))
