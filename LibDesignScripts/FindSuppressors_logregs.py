import pandas as pd
import numpy as np
import logomaker as lm
from pandas.api.types import CategoricalDtype
import math
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

aas=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']
grouped_aas = ['Q','N','S','T','E','D','K','R','H','Y','F','W','M','L','I','V','A','C','G','P','-']
FYN_core = [3, 17, 19, 25, 27, 49, 54]
FYN_wt = 'TLFVALYDYEARTEDDLSFHKGEKFQILNSSEGDWWEARSLTTGETGYIPSNYVAPV'

natural_DTS_cores = pd.read_csv('natural_DTS_cores.txt',sep='\t', index_col=0) ## Change to Detrimental_singles_FYN_in_structure_MSA.txt for suppressor mutation finding on detrimental single mutant queries
all_structure_MSA = pd.read_csv('FYN_SH3_clFoldseek_allstructureMSA_noindel.txt', sep='\t', index_col=0)
reliable_isofolds = pd.read_csv('reliable_isofolds.txt', sep='\t',index_col=0)

selected_coefficients = pd.DataFrame()
coef_iterator = 0

for index_core, row_core in natural_DTS_cores.iterrows():

    # Logoplot sequences with and without the query

    for query_status in ['with_query', 'without_query']:

        if query_status == 'with_query':
            structure_MSA = all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore']==index_core]['Target'])].copy()
        elif query_status == 'without_query':
            structure_MSA = all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore']!=index_core]['Target'])].copy()

        logo_df = pd.DataFrame(columns=aas)

        for pos in range(1, len(structure_MSA.columns) + 1, 1):
            df = structure_MSA[str(pos)].value_counts().rename_axis('aa').reset_index(name='counts')

            for index, row2 in df.iterrows():
                logo_df.at[pos, row2.aa] = int(row2.counts)
        logo_df = logo_df.fillna(0)

        prob_mat = lm.transform_matrix(logo_df, from_type='counts', to_type='information')
        logo = lm.Logo(prob_mat, figsize=(15, 3), color_scheme='dmslogo_funcgroup')
        for pos in FYN_core:
            logo.highlight_position(p=pos, color='gold', alpha=.5)

        plt.yticks(fontsize=12)
        plt.xlabel('position', fontsize=14)
        plt.ylabel('frequency', fontsize=14)

        if query_status == 'with_query':
            plt.title('Natural SH3s with '+ index_core +' core (Nham '+str( row_core.Nham) +'): ' + str(len(structure_MSA.index)))
            plt.savefig('Natural_SH3s_with_'+ index_core + '.png', dpi=300)
        elif query_status == 'without_query':
            plt.title('Natural SH3s without ' + index_core + ' core (Nham ' + str(row_core.Nham) + '): ' + str(
                len(structure_MSA.index)))
            plt.savefig('Natural_SH3s_without_' + index_core + '.png', dpi=300)



    # Build feature matrix

    all_structure_MSA_onehot = all_structure_MSA.fillna('-').copy()

    ## Select columns corresponding to sequence features present in less than 5% or more than 95% of the sequences

    drop_features = []
    for i in range(1, len(reliable_isofolds.iloc[0]['QuerySeq']), 1):
        freqs = all_structure_MSA_onehot[str(i)].value_counts(normalize=True).to_frame()
        for index, row in freqs.iterrows():
            if row['proportion'] >= 0.95 or row['proportion'] <= 0.05:
                drop_features.append(str(i) + '_' + row.name)

    ## Drop column(s) corresponding to the query position(s)

    for pos in FYN_core:
        try:
            all_structure_MSA_onehot = all_structure_MSA_onehot.drop(str(pos), axis=1)
        except KeyError:
            continue

    ## 1-hot encoding with all possible amino acids in each position

    cat_dtype = pd.CategoricalDtype(grouped_aas, ordered=True)

    all_structure_MSA_onehot = pd.get_dummies(all_structure_MSA_onehot.astype(cat_dtype))

    ## Drop features previously selected

    for i in (drop_features):
        try:
            all_structure_MSA_onehot = all_structure_MSA_onehot.drop(i, axis=1)
        except KeyError:
            continue

    ## Drop features representing the FYN wt sequence <- we want to pick up mutations that allow a certain position

    for i in enumerate(FYN_wt):
        try:
            all_structure_MSA_onehot = all_structure_MSA_onehot.drop(str(i[0] + 1) + '_' + i[1], axis=1)
        except KeyError:
            continue

    ## Run association tests for all str(pos)itions and discard features negatively associated with our query <- this will hopefully avoid negative coefficients in the regression

    association_df = pd.DataFrame()

    number_of_tests = 0
    for pos in range(1, len(FYN_wt) + 1, 1):
        for aa in list(all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore'] != index_core]['Target'])][
                           str(pos)].value_counts().index):
            contingency = pd.DataFrame(columns=[aa, 'not ' + aa])
            if aa in list(all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore'] == index_core]['Target'])][
                              str(pos)].value_counts().index):
                contingency.at['test', aa] = \
                all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore'] == index_core]['Target'])][
                    str(pos)].value_counts()[aa]
            else:
                contingency.loc['test', aa] = 0
            contingency.at['test', 'not ' + aa] = \
            all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore'] == index_core]['Target'])][
                str(pos)].value_counts()[[i for i in all_structure_MSA.loc[
                list(reliable_isofolds.loc[reliable_isofolds['MutCore'] == index_core]['Target'])][str(pos)].value_counts().index if
                                          i != aa]].sum()
            contingency.at['rest', aa] = \
            all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore'] != index_core]['Target'])][
                str(pos)].value_counts()[aa]
            contingency.at['rest', 'not ' + aa] = \
            all_structure_MSA.loc[list(reliable_isofolds.loc[reliable_isofolds['MutCore'] != index_core]['Target'])][
                str(pos)].value_counts()[[i for i in all_structure_MSA.loc[
                list(reliable_isofolds.loc[reliable_isofolds['MutCore'] != index_core]['Target'])][str(pos)].value_counts().index if
                                          i != aa]].sum()
            contingency.fillna(0)
            odd_ratio, pval = stats.fisher_exact(contingency)
            if pval < 1e-323:
                pval = 1e-323

            association_df.at[str(pos) + aa, 'pos'] = pos
            association_df.at[str(pos) + aa, 'wt_res'] = FYN_wt[pos - 1]
            association_df.at[str(pos) + aa, 'test_res'] = aa
            association_df.at[str(pos) + aa, 'pval'] = pval
            association_df.at[str(pos) + aa, 'minuslogpval'] = -1 * math.log(pval)
            association_df.at[str(pos) + aa, 'oddsr'] = odd_ratio

            number_of_tests += 1

    association_df = association_df.sort_values('oddsr', ascending=False)
    association_df.to_csv('Association_tests_for_query_'+index_core+ '_Nham_' + str(row_core.Nham) +'.txt',sep='\t')

    drop_features = []
    for index, row in association_df.iterrows():
        if row.oddsr < 1:
            drop_features.append(str(int(row.pos)) + '_' + row.test_res)

    for i in (drop_features):
        try:
            all_structure_MSA_onehot = all_structure_MSA_onehot.drop(i, axis=1)
        except KeyError:
            continue

    ## Change feature names to mutation interpretable names

    column_names_mutation = []
    for column in all_structure_MSA_onehot.columns:
        column_names_mutation.append(FYN_wt[int(column[:-2]) - 1] + str(column[:-2]) + column[-1])

    all_structure_MSA_onehot.columns = column_names_mutation

    ## Define target

    Y = reliable_isofolds.set_index('Target')['MutCore'] == index_core
    Y = Y.to_frame()
    Y['MutCore'] = Y['MutCore'].astype(int)

    ## shuffle data and split it into train & test set

    x = all_structure_MSA_onehot
    y = Y

    x, y = shuffle(x, y, random_state=3)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=3)

    ## Iterate LogReg models until finding a minimal one with at least 6 positive coefficients

    nr_pos_coefs = 0
    C = 0.002 ## Change to 1 for no lasso regularization
    while nr_pos_coefs < 6:
        cv = KFold(n_splits=10)
        classifier = LogisticRegression(C=C, penalty='l1', solver='liblinear')

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure()
        i = 0
        CV_coefs = pd.DataFrame()
        for train, test in cv.split(xtrain, ytrain):
            fold_logreg = classifier.fit(xtrain.reset_index(drop=True).iloc[train],
                                         ytrain.reset_index(drop=True).iloc[train])
            probas_ = fold_logreg.predict_proba(xtrain.reset_index(drop=True).iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve(ytrain.reset_index(drop=True).iloc[test], probas_[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            df = pd.DataFrame(fold_logreg.coef_[0], index=xtrain.columns)
            df = df.loc[df[0] != 0].sort_values(0, ascending=False)
            CV_coefs = pd.concat([CV_coefs, df], axis=1)
            i += 1

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='gray',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=1, linestyle='--', alpha=.8)

        logregCV = LogisticRegressionCV(Cs=[C], penalty='l1', solver='liblinear', cv=10).fit(xtrain, ytrain)
        # get probabilities for clf ## https://stackoverflow.com/questions/30051284/plotting-a-roc-curve-in-scikit-yields-only-3-points
        y_proba = logregCV.predict_proba(xtest)

        # ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(np.array(ytest['MutCore']), y_proba[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=1, color='k',
                 label=r'ROC on test set (AUC = %0.2f)' % (roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                 label='Chance', alpha=.8)

        plt.legend()

        CV_coefs['mean'], CV_coefs['std'] = CV_coefs.mean(axis=1), CV_coefs.std(axis=1)

        CV_coefs = CV_coefs.sort_values('mean', ascending=False)

        nr_pos_coefs = len(CV_coefs.loc[CV_coefs['mean'] > 0])
        C += 0.0005

        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('LogReg Model for query ' + index_core + ' (Nham ' + str(row_core.Nham) + '), positive coefficients: ' + str(nr_pos_coefs))

        if nr_pos_coefs >= 6:
            plt.savefig('LogReg_Query_' + index_core + '_Nham_' + str(row_core.Nham) + '.png', dpi=300)
            for index, row in CV_coefs.iterrows():
                if row['mean'] > 0:
                    selected_coefficients.at[coef_iterator, 'Query_core'] = index_core
                    selected_coefficients.at[coef_iterator, 'Permissivity_variant'] = row.name
                    selected_coefficients.at[coef_iterator, 'Permissivity_wt'] = row.name[0]
                    selected_coefficients.at[coef_iterator, 'Permissivity_pos'] = row.name[1:-1]
                    selected_coefficients.at[coef_iterator, 'Permissivity_mut'] = row.name[-1]
                    selected_coefficients.at[coef_iterator, 'Permissivity_coef'] = row['mean']
                    selected_coefficients.at[coef_iterator, 'Permissivity_coef_std'] = row['std']

                    coef_iterator+=1

selected_coefficients.to_csv('Permissivity_selected_vars_for_core_mutations.txt', sep='\t')





