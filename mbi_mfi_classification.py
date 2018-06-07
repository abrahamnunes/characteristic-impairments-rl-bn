# ==============================================================================
#   THIS FILE PERFORMS THE CLASSIFICATION USING MBI AND MFI STATISTICS
#
# ==============================================================================

import pandas as pd
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

from scipy import interp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# ==============================================================================
#   SET SOME PARAMETERS
# ==============================================================================

nsteps  = 10
rng     = np.random.RandomState(6634)
rng_lrc = np.random.RandomState(7432)

# ==============================================================================
#   IMPORT DATA
# 		X = ndarray((nsubjects, 20)) 
#			where X[,:10] are MBI values for 10 trials back 
#			and where X[:,10:] are MFI values for 10 trials back 
#		y = ndarray((nsubjects, 3))
#			each row is a one hot vector [class=hc, class=bed, class=bn] for 
#			the given subject
# ==============================================================================

X = 'YOUR FEATURES HERE'
y_hot = 'YOUR ONE HOT CLASS LABELS HERE'

# ==============================================================================
#   CLASSIFICATION
# ==============================================================================
c = 1
alpha = 0
lr_penalty='l1'
n_splits = int(N/4)
skf = StratifiedKFold(n_splits=n_splits)

AUC = []
COEFS_BN = []

mcfig, ax = plt.subplots(figsize=(6, 5))
ax.set_xlabel('False Positive Rate'); ax.set_xlim([0, 1])
ax.set_ylabel('True Positive Rate'); ax.set_ylim([0, 1])
ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), ls='--', c='k', lw=1.5)

TPR_PBE = []
TPR_BED = []
TPR_BN = []
TPR_HCBD = []
TPR_HCBN = []
TPR_BDBN = []

mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in skf.split(X, np.argmax(y_hot, axis=1)):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y_hot[train_index,:], y_hot[test_index,:]

    scale_x = StandardScaler().fit(X_train)
    X_train = scale_x.transform(X_train)
    X_test  = scale_x.transform(X_test)

    # --------------------------------------------------------------------------
    #   INITIALIZE CLASSIFIERS
    # --------------------------------------------------------------------------

    lrc_pbe = LogisticRegression(random_state=rng_lrc, penalty=lr_penalty, C=c)
    lrc_bed = LogisticRegression(random_state=rng_lrc, penalty=lr_penalty, C=c)
    lrc_bn  = LogisticRegression(random_state=rng_lrc, penalty=lr_penalty, C=c)

    # --------------------------------------------------------------------------
    #   TRAIN
    # --------------------------------------------------------------------------
    lrc_pbe.fit(X=X_train, y=y_train[:, 0])
    lrc_bed.fit(X=X_train, y=y_train[:, 1])
    lrc_bn.fit(X=X_train, y=y_train[:, 2])

    COEFS_BN.append(lrc_bn.coef_)

    # --------------------------------------------------------------------------
    #   TEST
    # --------------------------------------------------------------------------
    yhat_pbe = lrc_pbe.predict_proba(X=X_test)
    yhat_bed = lrc_bed.predict_proba(X=X_test)
    yhat_bn  = lrc_bn.predict_proba(X=X_test)

    # --------------------------------------------------------------------------
    #   MEASURE PERFORMANCE
    # --------------------------------------------------------------------------
    auc_pbe = roc_auc_score(y_true=y_test[:, 0], y_score=yhat_pbe[:,1])
    auc_bed = roc_auc_score(y_true=y_test[:, 1], y_score=yhat_bed[:,1])
    auc_bn = roc_auc_score(y_true=y_test[:, 2], y_score=yhat_bn[:,1])


    fpr_pbe, tpr_pbe, thresh_pbe = roc_curve(y_true=y_test[:, 0],
                                             y_score=yhat_pbe[:,1])
    fpr_bed, tpr_bed, thresh_bed = roc_curve(y_true=y_test[:, 1],
                                             y_score=yhat_bed[:,1])
    fpr_bn, tpr_bn, thresh_bn = roc_curve(y_true=y_test[:, 2],
                                          y_score=yhat_bn[:,1])

    TPR_PBE.append(interp(mean_fpr, fpr_pbe, tpr_pbe))
    TPR_BED.append(interp(mean_fpr, fpr_bed, tpr_bed))
    TPR_BN.append(interp(mean_fpr, fpr_bn, tpr_bn))

    ax.plot(fpr_pbe, tpr_pbe, c='b', alpha=alpha, lw=1.5)
    ax.plot(fpr_bed, tpr_bed, c='r', alpha=alpha, lw=1.5)
    ax.plot(fpr_bn, tpr_bn, c='g', alpha=alpha, lw=1.5)

    AUC.append([auc_pbe, auc_bed, auc_bn])

print('=========== AUC ===========')
AUC = np.array(AUC)
aucmean = np.round(np.mean(AUC, axis=0), 2)
aucstd = np.std(AUC, axis=0)
auclci = np.round(aucmean - 1.96*aucstd/np.sqrt(AUC.shape[0]), 2)
aucuci = np.round(aucmean + 1.96*aucstd/np.sqrt(AUC.shape[0]), 2)


# FINALIZE ROC CURVES
mean_tpr_pbe = np.mean(TPR_PBE, axis=0); mean_tpr_pbe[-1] = 1; mean_tpr_pbe[0]=0
mean_tpr_bed = np.mean(TPR_BED, axis=0); mean_tpr_bed[-1] = 1; mean_tpr_bed[0]=0
mean_tpr_bn  = np.mean(TPR_BN, axis=0); mean_tpr_bn[-1]   = 1; mean_tpr_bn[0]=0

ax.plot(mean_fpr, mean_tpr_pbe,
        c='#374E55FF', lw=2,
        label='HC-Rest AUC=' +
               str(aucmean[0]) + ' (' +
               str(auclci[0]) + '-' + str(aucuci[0]) + ')')

ax.plot(mean_fpr, mean_tpr_bed,
        c='#DF8F44FF', lw=2,
        label='BED-Rest AUC='+
               str(aucmean[1]) + ' (' +
               str(auclci[1]) + '-' + str(aucuci[1]) + ')')

ax.plot(mean_fpr, mean_tpr_bn,
        c='#00A1D5FF', lw=2,
        label='BN-Rest AUC='+
               str(aucmean[2]) + ' (' +
               str(auclci[2]) + '-' + str(aucuci[2]) + ')')

plt.legend()

plt.savefig('results/plots/look10back-prediction-cv.png',
            bbox_inches='tight',
            dpi=300)
plt.show()

mean_coef_bn = np.mean(np.array(COEFS_BN), axis=0).flatten()
std_coef_bn = (np.std(np.array(COEFS_BN),axis=0)/np.sqrt(n_splits)).flatten()
fig, ax = plt.subplots(nrows=2, figsize=(8, 4))
ax[0].set_title('Model-Based Index')
ax[0].bar(np.arange(10)+1,
          mean_coef_bn[:10],
          yerr=std_coef_bn[:10],
          color='#374E55FF',
          edgecolor='k',
          linewidth=1.5)
ax[0].set_ylabel('Coefficient')
ax[0].set_xticklabels([])
ax[0].set_xticks([])

ax[1].set_title('Model-Free Index')
ax[1].bar(np.arange(10)+1,
          mean_coef_bn[10:],
          yerr=std_coef_bn[10:],
          color='#DF8F44FF',
          edgecolor='k',
          linewidth=1.5)
ax[1].set_ylabel('Coefficient')
ax[1].set_xticks(np.arange(10)+1)
ax[1].set_xlabel('N Steps Back')
plt.tight_layout()
plt.savefig('results/plots/importance-lasso-bn.png',
            bbox_inches='tight',
            dpi=300)
plt.show()
