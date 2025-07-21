import Bio.PDB
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from operator import itemgetter
import freesasa
from Bio.PDB.ResidueDepth import ResidueDepth
import statistics
from sklearn.cluster import KMeans

def calc_residue_dist(residue_one, residue_two) :
    """Returns the minimum distance between any atom in two residues (except for backbone C, O and N)"""
    if 'CB' in residue_one and 'CB' in residue_two:
        all_residue_one_distances=[]
        for atom_one in residue_one:
            atom_one_type=str(atom_one)
            atom_one_type=atom_one_type[6]
            if atom_one_type != 'H' and atom_one != residue_one['N'] and atom_one != residue_one['C'] and atom_one != residue_one['O'] and atom_one != residue_one['CA']:
                all_atom_one_distances=[]
                for atom_two in residue_two:
                    atom_two_type=str(atom_two)
                    atom_two_type=atom_two_type[6]
                    if atom_two_type != 'H' and atom_two != residue_two['N'] and atom_two != residue_two['C'] and atom_two != residue_two['O'] and atom_two != residue_two['CA']:
                        diff_vector = atom_one.coord - atom_two.coord
                        all_atom_one_distances.append(np.sqrt(np.sum(diff_vector * diff_vector)))
                all_residue_one_distances.append(min(all_atom_one_distances))
        return min(all_residue_one_distances)

def calc_residue_dist_allatom(residue_one, residue_two) :
    """Returns the minimum distance between any atom in two residues (except for backbone C, O and N)"""
    all_residue_one_distances=[]
    all_residue_one_distances=[]
    for atom_one in residue_one:
        atom_one_type=str(atom_one)
        atom_one_type=atom_one_type[6]
        if atom_one_type != 'H':
            all_atom_one_distances=[]
            for atom_two in residue_two:
                atom_two_type=str(atom_two)
                atom_two_type=atom_two_type[6]
                if atom_two_type != 'H':
                    diff_vector = atom_one.coord - atom_two.coord
                    all_atom_one_distances.append(np.sqrt(np.sum(diff_vector * diff_vector)))
            all_residue_one_distances.append(min(all_atom_one_distances))
    return min(all_residue_one_distances)

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def calc_dist_matrix_allatom(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist_allatom(residue_one, residue_two)
    return answer

#Trim PDB
class ResSelect(Select):
    def accept_residue(self, res):
        if res.id[1] > start_res and res.id[1] <= end_res and res.parent.id == chain:
            return True
        else:
            return False

#Remove non-protein atoms
class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

pdbs = ['2l15','3mef']

target='MTGIVKWFNADKGFGFITPDDGSKDVFVHFSAIQNDGYKSLDEGQKVSFTIESGAKGPAAGNVTS'

cc_dict_all={}
cc_ext_dict_all={}

for pdbiterator in pdbs:

    pdb_code = pdbiterator
    pdb_filename = 'pdb' + pdb_code + '.ent'

    structure_root = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)

    cc_dict={}
    cc_ext_dict={}

    modelnr=0
    for j in structure_root:
        #grab whatever model and remove heteroatoms
        model = structure_root[modelnr]
        modelnr+=1
        chain = 'A'
        io = PDBIO()
        io.set_structure(model[chain])
        io.save(pdb_code + str(modelnr) + '_fix.pdb', NonHetSelect())

        #grab de fixed PDB
        pdb_filename = pdb_code + str(modelnr) + '_fix.pdb'

        structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
        model = structure[0]

        chain = 'A'

        #read sequences
        chains = {chain.id:seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()} #Read amino acid sequence
        resseqs = {chain.id:[residue.id[1] for residue in chain.get_residues()] for chain in structure.get_chains()} #Read sequence numbers

        seq = chains[chain]
        seq_nr=resseqs[chain]

        #Trim PDB
        start_res = seq.find(target)+seq_nr[0]-1
        print(start_res)
        end_res = start_res+len(target)
        print(end_res)
        print(len(target))

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_code + str(0) + '_fix.pdb', ResSelect())

        #grab de fixed PDB again
        pdb_filename = pdb_code + str(0) + '_fix.pdb'

        structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
        model = structure[0]

        chain = 'A'

        #calculate maps
        dist_matrix = calc_dist_matrix(model[chain], model[chain])
        contact_map = dist_matrix < 5.0

        dist_matrix_allatom = calc_dist_matrix_allatom(model[chain], model[chain])
        contact_map_allatom = dist_matrix_allatom < 5.0

        #read sequences
        #chains = {chain.id:seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()} #Read amino acid sequence
        resseqs = {chain.id:[residue.id[1] for residue in chain.get_residues()] for chain in structure.get_chains()} #Read sequence numbers

        #seq = chains[chain]
        #seq=list(seq)
        seq_nr=resseqs[chain]

        #Build contact network and calculate RSA
        G = nx.Graph()
        G.add_nodes_from(np.arange(0,len(model[chain]),1))

        labels={}
        node_sizes=[]
        for i in range(0, len(model[chain])):
            labels[i]=i+1
            node_sizes.append((sum(contact_map_allatom[i])-1)*50)
            res_contact_list = list(contact_map_allatom[i])
            res_contact_dist_list = list(dist_matrix_allatom[i])
            for (j,k,l) in zip(res_contact_list, range(0, len(model[chain])), res_contact_dist_list):
                if j == True:
                    G.add_edge(i,k,length=l)


        #Set node 'size' attributes
        for n, data in G.nodes(data=True):
            data['size'] = node_sizes[n]

        structure_rsa = freesasa.Structure(pdb_filename)
        result = freesasa.calc(structure_rsa)

        #Calculate per residue RSA - https://github.com/freesasa/freesasa-python/pull/14 | https://github.com/freesasa/freesasa-python/issues/3

        residueAreas1 = result.residueAreas()
        residueTotal1 = dict(map(lambda x: (x[0], x[1].relativeTotal), residueAreas1['A'].items()))

        freesasa_list=[]
        for key, value in residueTotal1.items():
            freesasa_list.append(value)

        #Color network nodes according to RSA
        norm = colors.Normalize(vmin=min(freesasa_list), vmax=max(freesasa_list))
        f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('RdYlGn'))

        def f2hex(f2rgb, f):
            rgb = f2rgb.to_rgba(f)[:3]
            return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])

        for (n,data), i in zip(G.nodes(data=True), freesasa_list):
            data['color'] = f2hex(f2rgb, i)

        G=nx.relabel_nodes(G,labels)

        #Amazing source of all that follows: https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python

        betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
        eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality
        katz_dict = nx.katz_centrality_numpy(G) # Run katz centrality | Regular katz_centrality fails with all atom contact networks, fix: https://stackoverflow.com/questions/43208737/using-networkx-to-calculate-eigenvector-centrality
        closeness_dict = nx.closeness_centrality(G)

        # Assign each to an attribute in your network
        nx.set_node_attributes(G, betweenness_dict, 'betweenness')
        nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
        nx.set_node_attributes(G, katz_dict, 'katz')
        nx.set_node_attributes(G, closeness_dict, 'closeness')

        #Generate centrality lists
        eigenvector_list=[]
        katz_list=[]
        betweenness_list=[]
        closeness_list=[]
        for (key, value), (key2, value2), (key3, value3), (key4, value4) in zip(eigenvector_dict.items(), katz_dict.items(), betweenness_dict.items(), closeness_dict.items()):
            eigenvector_list.append(value)
            katz_list.append(value2)
            betweenness_list.append(value3)
            closeness_list.append(value4)

        #Calculate residue depth using Biopython and MSMS
        #try:
        #rd = ResidueDepth(model)

        #depth_list=[]
        #for i in range(seq_nr[0], seq_nr[-1]+1):
        #    a=rd['A',(' ', i, ' ')]
        #    depth_list.append(float(a[0]))
        #except RuntimeError:
        #    pass

        #Normalize lists
        katz_list_norm=[float(i-min(katz_list))/float(max(katz_list)-min(katz_list)) for i in katz_list]
        closeness_list_norm=[float(i-min(closeness_list))/float(max(closeness_list)-min(closeness_list)) for i in closeness_list]
        freesasa_list_norm=[float(i-min(freesasa_list))/float(max(freesasa_list)-min(freesasa_list)) for i in freesasa_list]
        #depth_list_norm=[float(i-min(depth_list))/float(max(depth_list)-min(depth_list)) for i in depth_list]


        X = np.array(list(zip(closeness_list_norm, freesasa_list_norm)))
        kmeans = KMeans(n_clusters=6, n_init=1000, max_iter=100000).fit(X) # https://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/
        centroids = kmeans.cluster_centers_
        labels = kmeans.predict(X)


        s=np.argsort(centroids.T[1]) #find indices of sorted RSA values
        central_core=(labels==s[0])
        central_core=[i+int(seq_nr[0]) for i, x in enumerate(central_core) if x]
        external_core=(labels==s[1])
        external_core=[i+int(seq_nr[0]) for i, x in enumerate(external_core) if x]
        boundary_core=(labels==s[2])
        boundary_core=[i+int(seq_nr[0]) for i, x in enumerate(boundary_core) if x]
        boundary=(labels==s[3])
        boundary=[i+int(seq_nr[0]) for i, x in enumerate(boundary) if x]
        boundary_surface=(labels==s[4])
        boundary_surface=[i+int(seq_nr[0]) for i, x in enumerate(boundary_surface) if x]
        surface=(labels==s[5])
        surface=[i+int(seq_nr[0]) for i, x in enumerate(surface) if x]

        cc_index= [i - seq_nr[0] for i in central_core]
        ec_index= [i - seq_nr[0] for i in external_core]
        #print(cc_index)
        #print(contact_map)

        core_extended_net=[]
        for i in cc_index:
            #print('\n'+str(i+seq_nr[0]))
            for j in ec_index:
                if contact_map[i][j] == True:
                    #print(j+seq_nr[0])
                    if j+seq_nr[0] not in core_extended_net:
                        core_extended_net.append(j+seq_nr[0])

        cen_index= [i - seq_nr[0] for i in core_extended_net]
        cc_extended=[]
        for i in cen_index:
            n=0
            #print('\n'+str(i+seq_nr[0]))
            for j in cc_index:
                if contact_map[j][i] == True:
                    n+=1
            k=len(cc_index)*0.5
            if n > k:
                #print(i+seq_nr[0])
                cc_extended.append(i+seq_nr[0])

        cc_dict[modelnr]=central_core
        cc_ext_dict[modelnr]=cc_extended

    #Put all cc clusters together in a dictionary nest, where 1st level key is the pdb, 2nd level is the model and value is the list of residues in the cc cluster

    cc_dict_all[pdbiterator]=cc_dict
    cc_ext_dict_all[pdbiterator]=cc_ext_dict

import pandas as pd

cc_aa=[]
total_samples=0
for key1,value1 in cc_dict_all.items():
    for key,value in value1.items():
        total_samples+=1
        for a in value:
            if a not in cc_aa:
                cc_aa.append(a)

aa_df = pd.DataFrame(columns=['Residue', 'Residue_frequency'])

l=0
for i in cc_aa:
    k=0
    for key1,value1 in cc_dict_all.items():
        for key, value in value1.items():
            for a in value:
                if i==a:
                    k+=1
    residue_freq=k/total_samples
    aa_df.loc[l] = [i, round(residue_freq,2)]
    l+=1

aa_df=aa_df.sort_values('Residue_frequency', ascending=False)
aa_df.to_csv('Central_Core.txt', sep='\t')


cc_aa=[]
total_samples=0
for key1,value1 in cc_ext_dict_all.items():
    for key,value in value1.items():
        total_samples+=1
        for a in value:
            if a not in cc_aa:
                cc_aa.append(a)

aa_ext_df = pd.DataFrame(columns=['Residue', 'Residue_frequency'])

l=0
for i in cc_aa:
    k=0
    for key1,value1 in cc_ext_dict_all.items():
        for key, value in value1.items():
            for a in value:
                if i==a:
                    k+=1
    residue_freq=k/total_samples
    aa_ext_df.loc[l] = [i, round(residue_freq,2)]
    l+=1

aa_ext_df=aa_ext_df.sort_values('Residue_frequency', ascending=False)
aa_ext_df.to_csv('Extended_Core.txt', sep='\t')
