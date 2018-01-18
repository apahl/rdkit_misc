from collections import Counter
import base64

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
Draw.DrawingOptions.atomLabelFontFace = "DejaVu Sans"
Draw.DrawingOptions.atomLabelFontSize = 18

from rdkit.Chem.Draw import rdMolDraw2D

# from rdkit.Avalon import pyAvalonTools as pyAv

from Contrib.NP_Score import npscorer

try:
    # Try to import Avalon so it can be used for generation of 2d coordinates.
    from rdkit.Avalon import pyAvalonTools as pyAv
    USE_AVALON_2D = True
except ImportError:
    print("* Avalon not available. Using RDKit for 2d coordinate generation.")
    USE_AVALON_2D = False

from IPython.core.display import SVG

# import matplotlib.pyplot as plt
import matplotlib.colors as mpl_col

try:
    from cairosvg import svg2png
    PNG = True
except ImportError:
    PNG = False

# col_int = tuple(int(255 * x) for x in col[:3])
# ctag = "#{:02x}{:02x}{:02x}".format(col_int[0], col_int[1], col_int[2])
# HTML("""<p style='background-color:{};'>TEST</p>""".format(ctag))

CDICT = {
    "rwg": {  # colormap red-white-green
        "red": ((0.0, 1.0, 1.0),
                (0.5, 1.0, 1.0),
                (1.0, 0.5, 0.5),),
        "green": ((0.0, 0.5, 0.5),
                  (0.5, 1.0, 1.0),
                  (1.0, 1.0, 1.0),),
        "blue": ((0.0, 0.5, 0.5),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.5, 0.5),)
    },
}

fscore = npscorer.readNPModel()


class NormalizeAroundZero():
    """A class that when called normalizes the given values such that:
        values < 0 will be transformed to [0..0.5[ and
        values > 0 will become ]0.5..1.0]
        0 becomes 0.5"""

    def __init__(self, vmin, vmax):
        self.vmin = vmin
        if self.vmin > 0:
            self.vmin = 0
        self.vmax = vmax
        if self.vmax < 0:
            self.vmax = 0

    def __call__(self, val):
        if val < 0.0:
            if val <= self.vmin:
                return 0.0
            else:
                # val and self.vmin are negative !
                return 0.5 - (0.5 * val / self.vmin)
        elif val > 0.0:
            if val >= self.vmax:
                return 1.0
            else:
                return 0.5 + (0.5 * val / self.vmax)
        else:
            return 0.5


def check_2d_coords(mol, force=False):
    """Check if a mol has 2D coordinates and if not, calculate them."""
    if not force:
        try:
            mol.GetConformer()
        except ValueError:
            force = True  # no 2D coords... calculate them

    if force:
        if USE_AVALON_2D:
            pyAv.Generate2DCoords(mol)
        else:
            mol.Compute2DCoords()


def score_np(mol):
    return npscorer.scoreMol(mol, fscore)


def make_transparent(img):
    img = img.convert("RGBA")
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    return img


def mol_img_tag(svg_bc, options=None):
    tag = """<img {} src="data:image/png;base64,{}" alt="Mol"/>"""
    if options is None:
        options = ""
    img_tag = tag.format(options, base64.b64encode(svg_bc).decode())
    return img_tag


def highlight_np_scores(mol, col_map="rwg", output="svg", png_fn="contribs.png", width=400, height=200):
    """Fragment highlighting for Peter Ertls Natural Product Likeness Score
    (J Chem Inf Model. 2008 Jan;48(1):68-74; DOI: 10.1021/ci700286x),
    as implemented in the RDKit.
    output can be: `svg` (SVG image; default); `raw` (raw SVG string);
        `png` (PNG image, written to `png_fn`; requires cairoSVG);
        `png_tag` (HTML img tag containing the encoded PNG image);
        `debug` (text output of parameters).
    Helpful RDKit links (used for creating the code):
        http://www.rdkit.org/docs/GettingStartedInPython.html#explaining-bits-from-morgan-fingerprints
        http://rdkit.blogspot.de/2015/02/new-drawing-code.html"""
    output = output.lower()
    cmap = mpl_col.LinearSegmentedColormap(col_map, CDICT[col_map], 50)
    bit_info = {}
    rdMolDescriptors.GetMorganFingerprint(mol, 2, bitInfo=bit_info)
    num_atoms = 1  # float(mol.GetNumAtoms())
    atom_scores = Counter()
    bond_scores = Counter()
    num_bits = 0
    for bit in bit_info:
        if bit not in fscore:
            continue
        num_bits += 1
        score = fscore[bit]
        for frag in bit_info[bit]:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, frag[1], frag[0])
            for b_idx in env:
                bond_scores[b_idx] += (score / num_atoms)
                atom_scores[mol.GetBondWithIdx(
                            b_idx).GetBeginAtomIdx()] += (score / num_atoms)
                atom_scores[mol.GetBondWithIdx(
                            b_idx).GetEndAtomIdx()] += (score / num_atoms)

    if output == "debug":
        values = atom_scores.values()
        norm = NormalizeAroundZero(vmin=min(values), vmax=max(values))
        atom_cols = {atom: norm(score)
                     for atom, score in atom_scores.items()}
        values = bond_scores.values()
        norm = NormalizeAroundZero(vmin=min(values), vmax=max(values))
        bond_cols = {bond: norm(score)
                     for bond, score in bond_scores.items()}

        print("*** Scores ***:")
        print(atom_scores)
        print(bond_scores)
        print("*** Norm Scores ***:")
        print(atom_cols)
        print(bond_cols)
        return

    values = atom_scores.values()
    norm = NormalizeAroundZero(vmin=min(values), vmax=max(values))
    atom_cols = {atom: cmap(norm(score))
                 for atom, score in atom_scores.items()}
    values = bond_scores.values()
    norm = NormalizeAroundZero(vmin=min(values), vmax=max(values))
    bond_cols = {bond: cmap(norm(score))
                 for bond, score in bond_scores.items()}

    check_2d_coords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol, highlightAtoms=atom_cols.keys(), highlightAtomColors=atom_cols,
                        highlightBonds=bond_cols.keys(), highlightBondColors=bond_cols)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg = svg.replace("</svg>\n", "</svg>")
    # svg = svg.replace("\n", "")
    svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    if output == "raw":
        return svg
    elif "png" in output:
        if not PNG:
            print("Converting to PNG requires cairoSVG, which could not be found.\n"
                  "Try `pip install cairoSVG` to resolve this.")
            return
        if output == "png":
            svg2png(bytestring=svg, write_to=png_fn)
            return
        if output == "png_tag":  # return a HTML <img> tag containing the PNG img
            svg_bc = svg2png(bytestring=svg)
            return mol_img_tag(svg_bc)
        else:
            print("Unknown output option.")
            return
    elif output == "svg":
        return SVG(svg)
    else:
        print("Unknown output option:", output)
