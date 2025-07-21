from dogma import GeneticCode
from dogma.extensions import CodonSelector
import itertools
import pandas as pd
from argparse import ArgumentParser

#Define genetic code. Here, the standard genetic code (ncbi_id=1) with a glutamine amber suppression is used.
supE = GeneticCode(ncbi_id=1)

parser = ArgumentParser()
parser.add_argument('-aas', '--names-list', nargs='+', default=[])
aas = parser.parse_args().names_list

best_codons_by_group_df = pd.DataFrame(columns=['Target','Target_Num_uAAs','Labels', 'AAs', 'uAAs',
                                      'Num_Codons', 'Num_uAAs',
                                      'Stops', 'EquimolarAA'])

k=0
for x in range(2,len(aas)+1):
    equimolar=False
    best_codons_df = pd.DataFrame(columns=['Target','Target_Num_uAAs','Labels', 'AAs', 'uAAs',
                                      'Num_Codons', 'Num_uAAs',
                                      'Stops', 'EquimolarAA'])
    a=list(itertools.combinations(aas, x))
    l=0
    for i in a:
        must_include = i
        #Initialize CodonSelector object with list of one-letter amino acids to "must_exclude" (or "must_include")
        cs = CodonSelector(
        genetic_code=supE,
        must_include=must_include
        )
        df = cs.filtered_table.sort_values('Num_Codons', ascending=True)
        df=df.astype({"Stops":str,"EquimolarAA":str})
        best_codons_df.loc[l]=df.iloc[0]
        best_codons_df['Target'].loc[l]=''.join(i)
        best_codons_df['Target_Num_uAAs'].loc[l]=len(i)
        l += 1
    best_codons_df=best_codons_df.sort_values('Num_uAAs', ascending=True)
    best_codons_df=best_codons_df.reset_index(drop=True)
    #print(best_codons_df)
    for key, value in best_codons_df.iterrows():
        if value.Target_Num_uAAs == value.Num_uAAs:
            equimolar=True
            best_codons_by_group_df.loc[k]=value
            k+=1
    if equimolar==False:
        best_codons_by_group_df.loc[k]=best_codons_df.loc[0]
        k+=1


print(best_codons_by_group_df)
best_codons_by_group_df.to_csv('Codons_for_'+''.join(aas)+'.txt',sep='\t',index=False)