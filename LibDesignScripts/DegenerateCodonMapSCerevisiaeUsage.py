import operator
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-codon', '--names-list', nargs='+', default=[])
codon = parser.parse_args().names_list

SC_CU = {'Phe': {'TTT': 0.59, 'TTC': 0.41},
         'Leu': {'TTA': 0.2755, 'TTG': 0.2863, 'CTT': 0.1291, 'CTC': 0.0573, 'CTA': 0.1413, 'CTG': 0.1104},
         'Ile': {'ATT': 0.4629, 'ATC': 0.2637, 'ATA': 0.2734},
         'Met': {'ATG': 1.00},
         'Val': {'GTT': 0.39, 'GTC': 0.21, 'GTA': 0.21, 'GTG': 0.19},
         'Ser': {'TCT': 0.26, 'TCC': 0.16, 'TCA': 0.21, 'TCG': 0.1, 'AGT': 0.16, 'AGC': 0.11},
         'Pro': {'CCT': 0.31, 'CCC': 0.15, 'CCA': 0.42, 'CCG': 0.12},
         'Thr': {'ACT': 0.345, 'ACC': 0.22, 'ACA': 0.295, 'ACG': 0.14},
         'Ala': {'GCT': 0.38, 'GCC': 0.22, 'GCA': 0.29, 'GCG': 0.11},
         'Tyr': {'TAT': 0.56, 'TAC': 0.44},
         '*': {'TAA': 0.47, 'TAG': 0.23, 'TGA': 0.3},
         'His': {'CAT': 0.64, 'CAC': 0.36},
         'Gln': {'CAA': 0.69, 'CAG': 0.31},
         'Asn': {'AAT': 0.59, 'AAC': 0.41},
         'Lys': {'AAA': 0.58, 'AAG': 0.42},
         'Asp': {'GAT': 0.65, 'GAC': 0.35},
         'Glu': {'GAA': 0.7, 'GAG': 0.3},
         'Cys': {'TGT': 0.63, 'TGC': 0.37},
         'Trp': {'TGG': 1},
         'Arg': {'CGT': 0.14, 'CGC': 0.06, 'CGA': 0.07, 'CGG': 0.04, 'AGA': 0.48, 'AGG': 0.21},
         'Gly': {'GGT': 0.47, 'GGC': 0.19, 'GGA': 0.22, 'GGG': 0.12}}

IUPAC_nucl_code = {'A': 'A',
                   'T': 'T',
                   'G': 'G',
                   'C': 'C',
                   'R': ['A', 'G'],
                   'Y': ['C', 'T'],
                   'S': ['G', 'C'],
                   'W': ['A', 'T'],
                   'K': ['G', 'T'],
                   'M': ['A', 'C'],
                   'B': ['C', 'G', 'T'],
                   'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'],
                   'V': ['A', 'C', 'G'],
                   'N': ['A', 'T', 'G', 'C']}

deg_codon = list(codon[0])

pos1 = []
pos2 = []
pos3 = []
for i in deg_codon:
    for key, value in IUPAC_nucl_code.items():
        if i == key and not pos1:
            pos1 = value
        elif i == key and not pos2:
            pos2 = value
        elif i == key and not pos3:
            pos3 = value

unfolded_codon = []
for i in pos1:
    for j in pos2:
        for k in pos3:
            unfolded_codon.append(i + j + k)

#print(unfolded_codon)

SC_CU_codon = {}
for d in unfolded_codon:
    for key, value in SC_CU.items():
        for key2, value2 in value.items():
            if d == key2:
                SC_CU_codon[key] = value

for key, value in SC_CU_codon.items():
    sorted_values = sorted(value.items(), key=operator.itemgetter(1))

    keys = []
    values = []
    for i in sorted_values:
        keys.append(i[0])
        values.append(i[1])

    sorted_dictionary = dict(zip(keys, values))

    SC_CU_codon[key] = sorted_dictionary

SC_CU_df = pd.DataFrame(columns=['AA', 'Codon', 'Freq', 'Color'])

l = 0
for key, value in SC_CU_codon.items():
    k = 0
    for key2, value2 in value.items():
        b = []
        b.append(key)
        b.append(key2)
        b.append(value2)
        if key2 in unfolded_codon:
            b.append('red')
        else:
            n = int((1 - value2) * 256)
            b.append('#%02x%02x%02x' % (n, n, n))
        k += 1
        SC_CU_df.loc[l] = b
        l += 1

SC_CU_df = SC_CU_df.sort_values('Freq', ascending=True)
SC_CU_df = SC_CU_df.reset_index(drop=True)

l = 0
for d in SC_CU_df.iterrows():
    if l < 10:
        SC_CU_df['Codon', l] = '0' + str(l) + SC_CU_df['Codon'].loc[l]
    else:
        SC_CU_df['Codon', l] = str(l) + SC_CU_df['Codon'].loc[l]
    l += 1

SC_CU_df = SC_CU_df.sort_values('Codon')
colors = SC_CU_df.Color.tolist()

# print(SC_CU_df)
# print(SC_CU_df.set_index(['AA','Codon']).unstack()['Freq'])

ax = SC_CU_df.set_index(['AA', 'Codon']).unstack()['Freq'].plot.bar(stacked=True, figsize=(15, 8), legend=False,
                                                                    color=colors, edgecolor='black', lw=2, fontsize=16)

plt.xlabel('Coded Amino Acid', fontsize=18)
plt.ylabel('Codon usage \ Freq', fontsize=18)
plt.title(''.join(deg_codon) + ' codon mapped on S. cerevisiae codon usage', fontsize=20)

SC_CU_df_plot = SC_CU_df.set_index(['AA', 'Codon']).unstack()['Freq']

heights = []
widths = []
labels_x = []
labels_y = []
for rect in ax.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()
    label_x = x + width / 2
    label_y = y + height / 2

    if height > 0:
        heights.append(height)
        widths.append(width)
        labels_x.append(label_x)
        labels_y.append(label_y)

for height, width, label_x, label_y, name in zip(heights, widths, labels_x, labels_y, SC_CU_df_plot.columns):
    label_text = name[-3:] + '\n' + f'{round(height, 2)}'
    ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)

plt.savefig(''.join(deg_codon) + '.png')