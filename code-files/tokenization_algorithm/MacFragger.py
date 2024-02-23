# RDKit: A collection of cheminformatics and machine learning tools.
from rdkit import Chem
# Chem.BRICS: A module in RDKit for breaking bonds based on BRICS rules.
from rdkit.Chem.BRICS import BreakBRICSBonds
# RDLogger: Used to suppress RDKit warnings for cleaner output.
from rdkit import RDLogger
# itertools.combinations: For generating combinations of elements.
from itertools import combinations
# igraph: A library for creating and manipulating undirected graphs, used here to manage molecular fragments as graph components.
from igraph import Graph
import copy
import argparse

RDLogger.DisableLog('rdApp.*')

class MacFragGenerator:
    # __init__ Method: Initializes the class with predefined chemical environments (self.environs) and reaction definitions (self.reactionDefs) 
    # used to identify potential bonds to break in a molecule. 
    # These are based on SMARTS patterns representing different chemical groups.
    def __init__(self):
        self.environs = {
            'L1': '[C;D3]([#0,#6,#7,#8])(=O)',  #original L1
            'L2': '[O;D2]-[#0,#6,#1]',  #original L3
            'L3': '[C;!D1;!$(C=*)]-[#6]',  #original L4
            'L4': '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',  #original L5
            'L5': '[C;D2,D3]-[#6]',  #original L7
            'L6': '[C;!D1;!$(C!-*)]',  #original L8
            'L61': '[C;R1;!D1;!$(C!-*)]',
            'L7': '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]',  #original L9
            'L8': '[N;R;$(N(@C(=O))@[#6,#7,#8,#16])]',  #original L10
            'L9': '[S;D2](-[#0,#6])',  #original L11
            'L10': '[S;D4]([#6,#0])(=O)(=O)',  #original L12
            
            'L11': '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',  #original L13
            'L111': '[C;R2;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
            'L112': '[C;R1;$(C(-;@[C,N,O,S;R2])-;@[N,O,S;R2])]',
            
            'L12': '[c;$(c(:[c,n,o,s]):[n,o,s])]',  #original L14
            
            'L13': '[C;$(C(-;@C)-;@C)]',  #original L15
            'L131': '[C;R2;$(C(-;@C)-;@C)]',
            'L132': '[C;R1;$(C(-;@[C;R2])-;@[C;R2])]',
            
            'L14': '[c;$(c(:c):c)]', #original L16
        }
        
        self.reactionDefs = [
        # L1
        [('1', '2', '-'),
            ('1', '4', '-'),
            ('1', '8', '-'),
            ('1', '11', '-'),
            ('1', '12', '-'),
            ('1', '13', '-'),
            ('1', '14', '-')],

        # L2
            [('2', '3', '-'),
            ('2', '11', '-'),
            ('2', '12', '-'),
            ('2', '13', '-'),
            ('2', '14', '-')],

        # L3
            [('3', '4', '-'),
            ('3', '9', '-')],

        # L4
            [('4', '10', '-'),
            ('4', '12', '-'),
            ('4', '14', '-'),
            ('4', '11', '-'),
            ('4', '13', '-')],

        # L5
            [('5', '5', '=')],

        # L6
            [('6', '7', '-'),
            ('6', '8', '-'),
            ('6', '11', '-;!@'),
            ('6', '12', '-'),
            ('6', '13', '-;!@'),
            ('6', '14', '-')],
            
        # L61
            [('61', '111', '-;@'),
            ('61', '131', '-;@')],   

        # L7
            [('7', '11', '-'),  
            ('7', '12', '-'),  
            ('7', '13', '-'),
            ('7', '14', '-')],

        # L8
            [('8', '11', '-'),
            ('8', '12', '-'),
            ('8', '13', '-'),
            ('8', '14', '-')],

        # L9
            [('9', '11', '-'),
            ('9', '12', '-'),
            ('9', '13', '-'),
            ('9', '14', '-')],

        # L11
            [('11', '12', '-'),
            ('11', '13', '-;!@'),
            ('11', '14', '-')],
            
        # L112
            [('112', '132', '-;@')],

        # L12
            [('12', '12', '-'),  
            ('12', '13', '-'),
            ('12', '14', '-')],

        # L13
            [('13', '14', '-')],

        # L14
            [('14', '14', '-')],  
        ]
        
        self.environMatchers = {}
        for env, sma in self.environs.items():
            self.environMatchers[env] = Chem.MolFromSmarts(sma)

        self.bondMatchers = []
        for compats in self.reactionDefs:
            tmp = []
            for i1, i2, bType in compats:
                e1 = self.environs['L%s' % i1]
                e2 = self.environs['L%s' % i2]
                patt = '[$(%s)]%s[$(%s)]' % (e1, bType, e2)
                patt = Chem.MolFromSmarts(patt)
                tmp.append((i1, i2, bType, patt))
            self.bondMatchers.append(tmp)
    
    def SSSRsize_filter(self, bond, maxSR=8):
        judge = True
        for i in range(3, maxSR + 1):
            if bond.IsInRingSize(i):
                judge = False
                break
        return judge
        
    def searchBonds(self, mol, maxSR=8):
        bondsDone = set()
        envMatches = {}   
        for env, patt in self.environMatchers.items():
            envMatches[env] = mol.HasSubstructMatch(patt)
        for compats in self.bondMatchers:
            for i1, i2, bType, patt in compats:
                if not envMatches['L' + i1] or not envMatches['L' + i2]:
                    continue
                
                #Convert the patterns to SMARTS


                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    if match not in bondsDone and (match[1], match[0]) not in bondsDone:
                        bond = mol.GetBondBetweenAtoms(match[0], match[1])
                        
                        if not bond.IsInRing():
                            bondsDone.add(match) 
                            yield (((match[0], match[1]), (i1, i2)))
                        elif bond.IsInRing() and self.SSSRsize_filter(bond, maxSR=maxSR):
                            bondsDone.add(match)
                            yield (((match[0], match[1]), (i1, i2)))
        
    def mol_with_atom_index(self, mol):
        # ... implementation ...
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return mol
    
    def mol_remove_atom_mapnumber(self, mol):
        # ... implementation ...
        for atom in mol.GetAtoms():
            atom.ClearProp('molAtomMapNumber')
        return mol
    
    def Get_block_index(self, blocks):
        # ... implementation ...
        block_index = {}
        i = 0
        for bs in blocks:
            tmp = [a.GetAtomMapNum() for a in bs.GetAtoms() if a.GetSymbol() != '*']
            block_index[tuple(tmp)] = i
            i += 1      
        return block_index
    
    def extrac_submol(self, mol, atomList, link):
        aList_mol = list(range(mol.GetNumAtoms()))
        aList_link = list(set([a[0] for a in link]))
        aList_submol = list(set(atomList + aList_link))
        aList_remove = [a for a in aList_mol if a not in aList_submol]
        eMol = Chem.RWMol(mol)
        
        aList_bbond = [a for a in aList_link if a not in atomList]
        for b in combinations(aList_bbond, 2):
            eMol.RemoveBond(b[0], b[1])
        
        aList_remove.sort(reverse=True)
        for a in aList_remove:
            eMol.RemoveAtom(a) 
        for ba, btype in link:
            if ba in atomList:
                continue
            tmpatom = [a for a in eMol.GetAtoms() if a.GetAtomMapNum() == ba][0]
            tmpatom.SetIsAromatic(False)
            tmpatom.SetAtomicNum(0)
            tmpatom.SetIsotope(int(btype))
            tmpatom.SetNoImplicit(True)
                
        frag = eMol.GetMol()
    
        for a in frag.GetAtoms():
            a.ClearProp('molAtomMapNumber')   
        return frag
    
    def simple_iter(self, graph, k):
        # ... implementation ...
        cis = []
        colors = graph.vcount() * [-1]
        for i in range(graph.vcount() - 1, -1, -1):
            subgraph_set = [i]
            cis.append(copy.deepcopy(subgraph_set))
            colors[i] = 0
            extension = []
            ex_neighs = [[]]
            pointers = [extension.__len__() - 1]
            poi_col = [1]
            sub_size = 1
            while subgraph_set != []:
                while pointers[-1] > -1:
                    last = pointers[-1]
                    vertex = extension[last]
                    ver_col = colors[vertex]
                    pointers[-1] -= 1
                    act_vertex = pointers[-1]
                    if ver_col == sub_size:
                        extension.pop()
                    if act_vertex > -1:
                        if colors[extension[act_vertex]] < poi_col[-1]:
                            pointers[-1] = pointers[poi_col[-1] - 2]
                            poi_col[-1] = colors[extension[pointers[-1]]]
                    subgraph_set.append(vertex)
                    cis.append(copy.deepcopy(subgraph_set))
                    sub_size += 1
                    if sub_size == k:
                        subgraph_set.pop()
                        sub_size -= 1
                    else:
                        ex_neighs.append([])
                        found = False
                        for neig in graph.neighbors(vertex):
                            if neig >= i:
                                break
                            if colors[neig] == -1:
                                colors[neig] = sub_size
                                ex_neighs[-1].append(neig)
                                extension.append(neig)
                                found = True
                        if found == True:
                            pointers.append(extension.__len__() - 1)
                            poi_col.append(sub_size)
                        else:
                            pointers.append(pointers[-1])
                            poi_col.append(poi_col[-1])
                pointers.pop()
                poi_col.pop()
                subgraph_set.pop()
                for vertex in ex_neighs[-1]:
                    colors[vertex] = -1
                ex_neighs.pop()
                sub_size -= 1
            colors[i] = -1
        return cis       

    
    def MacFrag(self, mol, maxBlocks=6, maxSR=8, asMols=False, minFragAtoms=1):
        # ... implementation ...
        fragPool = {}
        mol = self.mol_with_atom_index(mol)   
        bonds = list(self.searchBonds(mol, maxSR=maxSR))
        fragmol = BreakBRICSBonds(mol, bonds=bonds)
        blocks = Chem.GetMolFrags(fragmol, asMols=True)
        for block in blocks:
            tmp = copy.deepcopy(block)
            tmp_smiles = Chem.MolToSmiles(tmp)  
            nAtoms = tmp.GetNumAtoms(onlyExplicit=True)
            if nAtoms - tmp_smiles.count('*') >= minFragAtoms:
                #handle the dummy atoms or wildcards atoms
                #Carbon (C): Atomic number 6
                #Oxygen (O): Atomic number 8
                #Nitrogen (N): Atomic number 7
                #Sulfur (S): Atomic number 16
                # for a in tmp.GetAtoms():
                #     if a.GetAtomicNum() == 0:  # Wildcard atom or dummy atom
                #         a.SetAtomicNum(6)  # Replace with carbon atom
                for a in tmp.GetAtoms():
                    a.ClearProp('molAtomMapNumber') 
                fragPool[tmp] = Chem.MolToSmiles(tmp)

        if maxBlocks > len(blocks):
            maxBlocks = len(blocks)
        if maxBlocks == 1:
            if asMols:
                return fragPool.keys()
            else:
                return list(set(fragPool.values()))
        
        block_index = self.Get_block_index(blocks)
        index_block = {block_index[key]: key for key in block_index.keys()}
                    
        bond_block = {}
        point = False
        block_link = {block: set() for block in block_index.keys()}
        for b in bonds:
            ba1, ba2 = b[0]
            btype1, btype2 = b[1]
            for block in block_index.keys():
                if ba1 in block:
                    bond_block[ba1] = block
                    block_link[block].add((ba2, btype2))
                    point = True
                if ba2 in block:
                    bond_block[ba2] = block
                    block_link[block].add((ba1, btype1))
                    point = True
                if point == True:
                    continue
        
        n = len(index_block.keys())    
        edges = []
        for b in bonds:
            ba1, ba2 = b[0]
            edges.append((block_index[bond_block[ba1]], block_index[bond_block[ba2]]))
        
        graph = Graph(n=n, edges=edges, directed=False)
        all_cis = self.simple_iter(graph, k=maxBlocks)
        for i in range(n):
            all_cis.remove([i])
        
        sub_link = {}
        for cis in all_cis:
            tmp = []
            tmp2 = set()
            for ni in cis:
                tmp.extend(list(index_block[ni]))
                tmp2 = tmp2.union(block_link[index_block[ni]])
            sub_link[tuple(tmp)] = tmp2  

        for fa in sub_link:
            frag = self.extrac_submol(mol, atomList=list(fa), link=sub_link[fa])
            frag_smiles = Chem.MolToSmiles(frag)    
            nAtoms = frag.GetNumAtoms(onlyExplicit=True)
            if nAtoms - frag_smiles.count('*') >= minFragAtoms:
                fragPool[frag] = frag_smiles
                    
        if asMols:
            return list(fragPool.keys())
        else:
            return list(set(fragPool.values()))

    
    def write_file(self, input_file, dir, maxBlocks, maxSR, asMols, minFragAtoms):
        # ... implementation ...
        non_mols = 0
        processed_mols = 0
        if '.smi' in input_file:
            smiles = [line.strip().split()[0] for line in open(input_file).readlines()]
            mols = [Chem.MolFromSmiles(s) for s in smiles]
        elif '.sdf' in input_file:
            mols = Chem.SDMolSupplier(input_file)
        else:
            print('Error: the input file should be .smi or .sdf file')
            
        if asMols == True:
            out_file = dir + '_fragments.sdf'
            fw = Chem.SDWriter(out_file)
            for mol in mols:
                if mol is not None:
                    processed_mols += 1
                    frags = self.MacFrag(mol, maxBlocks=maxBlocks, maxSR=maxSR, asMols=True, minFragAtoms=minFragAtoms)
                    for f in frags:
                        try:
                            fw.write(f)
                        except Exception:
                            continue
                else:
                    non_mols += 1
                    print('Error: the input file contains non-mols')
            
        elif asMols == False:
            out_file = dir  + '_mac_fragments'
            fw = open(out_file, 'w')
            for mol in mols:
                if mol is not None:
                    processed_mols += 1
                    frags = self.MacFrag(mol, maxBlocks=maxBlocks, maxSR=maxSR, asMols=False, minFragAtoms=minFragAtoms)
                    frags_smiles = [ f for f in frags]
                    frags_line = ','.join(frags_smiles)
                    fw.write(frags_line + '\n')
                else:
                    non_mols += 1
                    # convert mol to smiles
                    print('Error: the input file contains non-mols')
            fw.close()
        print('non_mols: ', non_mols)
        print('processed_mols: ', processed_mols)
  