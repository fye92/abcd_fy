from scipy import stats
import random
from utils import *
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


def train_cpm(train_mat, train_behav, num_nodes, p_thresh=0.05, mode='linear'):
    """
    Train CPM model given a training set.
    :param train_mat: flattened training feature matrix of size (v*(v-1)/2, n), where v is the number of nodes, and n is the number of subjects.
    :type train_mat: NumPy Array.
    :param train_behav: training behavioral data of size (n,).
    :type train_behav: NumPy Array.
    :param num_nodes: number of nodes in the network.
    :type num_nodes: integer.
    :param p_thresh: p threshold for determining significant edges.
    :type p_thresh: float.
    :param mode: what function to use when determining edges. 'linear' or 'ridge'.
    :type mode: string
    :return:
        pos_estimator: trained sklearn estimator using positive edges only.
        neg_estimator: trained sklearn estimator using negative edges only.
        both_estimator: trained sklearn estimator using both edges.
        pos_edges: (v, v) matrix with significant positive edges set to 1 and 0 otherwise.
        neg_edges: (v, v) matrix with significant negative edges set to 1 and 0 otherwise.
    :rtype: (sklearn estimator, sklearn estimator, sklearn estimator, NumPy Array, NumPy Array)
    """
    corr_train = [stats.pearsonr(train_behav, mat) for mat in train_mat]
    r_lst = np.array([c[0] for c in corr_train])
    p_lst = np.array([c[1] for c in corr_train])
    # check
    print('Number of np.nan in r_lst: {}'.format(np.count_nonzero(np.isnan(r_lst))))

    r_mat = np.zeros((num_nodes, num_nodes))
    p_mat = np.zeros((num_nodes, num_nodes))
    iu = np.triu_indices(num_nodes, k=1)  # upper triangle index
    il = np.tril_indices(num_nodes, k=-1)  # lower triangle index
    r_mat[iu] = r_lst
    p_mat[iu] = p_lst
    r_mat[il] = r_mat.T[il]
    p_mat[il] = p_mat.T[il]
    np.fill_diagonal(r_mat, np.nan)
    np.fill_diagonal(p_mat, np.nan)

    print(">> Checking symmetry...")
    if check_symmetric(r_mat) and check_symmetric(p_mat):
        print(">>> Passed.")
    else:
        sys.exit(">>> ERROR: r_mat or p_mat not symmetric. Please check your data.")

    pos_edges = (r_mat > 0) & (p_mat < p_thresh)
    pos_edges = pos_edges.astype(int)
    neg_edges = (r_mat < 0) & (p_mat < p_thresh)
    neg_edges = neg_edges.astype(int)

    pos_sum = train_mat[pos_edges[iu].astype(bool), :]
    neg_sum = train_mat[neg_edges[iu].astype(bool), :]
    pos_sum = pos_sum.sum(axis=0)  # sum of weights of selected edges
    neg_sum = neg_sum.sum(axis=0)
    both = pos_sum - neg_sum

    if mode == 'ridge':
        alphas = 10 ** np.linspace(7, -3, 51) * 0.5

    num_pos_edge = sum(pos_edges[iu])
    num_neg_edge = sum(neg_edges[iu])
    num_total_edge = num_neg_edge + num_pos_edge

    if num_pos_edge != 0:  # if positive edges are identified
        print(">> Found {} significant positive edges based on p threshold of {}".format(num_pos_edge, p_thresh))
        if mode == 'linear':
            pos_estimator = LinearRegression(fit_intercept=True, normalize=False)
        elif mode == 'ridge':
            pos_estimator = GridSearchCV(Ridge(fit_intercept=True, normalize=False), param_grid={'alpha': alphas}, cv=10)
        elif mode == 'logistic':
            pos_estimator = LogisticRegression()
        else:
            print("ERROR: mode {} not implemented!".format(mode))
            quit()
        pos_estimator.fit(pos_sum.reshape(-1, 1), train_behav.reshape(-1, 1))  # reshape: num_subjects x num_features
    else:
        print(">> WARNING: Zero significant positive edges identified.")
        pos_estimator = np.nan

    if num_neg_edge != 0:  # if negative edges are identified
        print(">> Found {} significant negative edges based on p threshold of {}".format(num_neg_edge, p_thresh))
        if mode == 'linear':
            neg_estimator = LinearRegression(fit_intercept=True, normalize=False)
        elif mode == 'ridge':
            neg_estimator = GridSearchCV(Ridge(fit_intercept=True, normalize=False), param_grid={'alpha': alphas}, cv=10)
        elif mode == 'logistic':
            neg_estimator = LogisticRegression()
        else:
            print("ERROR: mode {} not implemented!".format(mode))
            quit()
        neg_estimator.fit(neg_sum.reshape(-1, 1), train_behav.reshape(-1, 1))
    else:
        print(">> WARNING: Zero significant negative edges identified.")
        neg_estimator = np.nan

    if num_total_edge != 0:  # if significant edges are identified
        print(">> Found {} significant edges based on p threshold of {}".format(num_total_edge, p_thresh))
        if mode == 'linear':
            both_estimator = LinearRegression(fit_intercept=True, normalize=False)
        elif mode == 'ridge':
            both_estimator = GridSearchCV(Ridge(fit_intercept=True, normalize=False), param_grid={'alpha': alphas}, cv=10)
        elif mode == 'logistic':
            both_estimator = LogisticRegression()
        else:
            print("ERROR: mode {} not implemented!".format(mode))
            quit()
        both_estimator.fit(both.reshape(-1, 1), train_behav.reshape(-1, 1))
    else:
        print(">> WARNING: Zero significant edges identified.")
        both_estimator = np.nan

    return pos_estimator, neg_estimator, both_estimator, pos_edges, neg_edges


def kfold_cpm(x, y, k, p_thresh=0.05, zscore=False, mode='linear'):
    """
    Run CPM using k-fold cross-validation.
    :param x: stacked feature matrix of size (v, v, n) where v is the number of nodes, and n is the number of subjects.
    :type x: NumPy Array.
    :param y: behavioral data of size (n,).
    :type y: NumPy Array.
    :param k: number of folds.
    :type k: integer.
    :param p_thresh: p threshold for determining significant edges.
    :type p_thresh: float.
    :param zscore: whether to z-score edge strength in the training set.
    :type zscore: bool
    :param mode: what function to use when determining edges. 'linear' or 'ridge'.
    :type mode: string
    :return:
        y_pred_pos: an array of size (n,) containing predicted behav scores for all subjects during testing using positive edges.
        y_pred_neg: an array of size (n,) containing predicted behav scores for all subjects during testing using negative edges.
        y_pred_both: an array of size (n,) containing predicted behav scores for all subjects during testing using both edges.
        fit_p: each entry contains sklearn estimator (or np.nan) obtained from the training set of that fold using positive edges.
        fit_n: each entry contains sklearn estimator (or np.nan) obtained from the training set of that fold using negative edges.
        fit_b: each entry contains sklearn estimator (or np.nan) obtained from the training set of that fold using both edges.
        edges_p: each matrix is (v, v) and contains 1 for significant positive edges and 0 otherwise, identified from the training set of that fold.
        edges_n: each matrix is (v, v) and contains 1 for significant negative edges and 0 otherwise, identified from the training set of that fold.
    :rtype: (NumPy Array, NumPy Array, NumPy Array, sklearn estimator, sklearn estimator, sklearn estimator, list, list)
    """
    num_subs = x.shape[2]
    num_nodes = x.shape[0]
    iu = np.triu_indices(num_nodes, k=1)  # upper triangle index
    all_edges = np.zeros([int(num_nodes * (num_nodes - 1) / 2), num_subs])
    for i in range(num_subs):
        all_edges[:, i] = x[:, :, i][iu]

    x = all_edges  # shape should be (v*(v-1)/2, n)
    rand_inds = np.arange(0, num_subs)
    random.shuffle(rand_inds)

    sample_size = int(np.floor(float(num_subs) / k))

    y_pred_pos = np.zeros(num_subs)
    y_pred_neg = np.zeros(num_subs)
    y_pred_both = np.zeros(num_subs)
    fit_p = []
    fit_n = []
    fit_b = []
    edges_p = []
    edges_n = []

    for fold in range(0, k):
        si = fold * sample_size
        fi = (fold + 1) * sample_size

        if fold != k - 1:
            test_inds = rand_inds[si:fi]
        else:
            test_inds = rand_inds[si:]

        train_inds = rand_inds[~np.isin(rand_inds, test_inds)]

        train_mats = x[:, train_inds]
        train_behav = y[train_inds]

        test_mats = x[:, test_inds]

        if zscore:
            scaler = preprocessing.StandardScaler().fit(train_mats.T)
            train_mats = scaler.transform(train_mats.T).T
            test_mats = scaler.transform(test_mats.T).T

        pos_estimator, neg_estimator, both_estimator, pos_edges, neg_edges = train_cpm(train_mats, train_behav, num_nodes, p_thresh, mode)

        edges_p.append(pos_edges)
        edges_n.append(neg_edges)
        fit_p.append(pos_estimator)
        fit_n.append(neg_estimator)
        fit_b.append(both_estimator)

        pos_sum = np.sum(test_mats[pos_edges[iu].astype(bool), :], axis=0)
        neg_sum = np.sum(test_mats[neg_edges[iu].astype(bool), :], axis=0)
        both = pos_sum - neg_sum

        if not isinstance(pos_estimator, float):  # if a fit was successfully obtained, i.e. est not np.nan
            y_pred_pos[test_inds] = pos_estimator.predict(pos_sum.reshape(-1, 1)).flatten()
        else:
            y_pred_pos[test_inds] = np.nan

        if not isinstance(neg_estimator, float):
            y_pred_neg[test_inds] = neg_estimator.predict(neg_sum.reshape(-1, 1)).flatten()
        else:
            y_pred_neg[test_inds] = np.nan

        if not isinstance(both_estimator, float):
            y_pred_both[test_inds] = both_estimator.predict(both.reshape(-1, 1)).flatten()
        else:
            y_pred_both[test_inds] = np.nan

    return y_pred_pos, y_pred_neg, y_pred_both, fit_p, fit_n, fit_b, edges_p, edges_n


def run_cpm_thread(y, iter, x, k, out_path, p_thresh=0.05, zscore=False, mode='linear'):
    """
    Called by run_cpm.py to run cpm in parallel. Do not change variable ordering or partial won't work.
    :param x: stacked feature matrix of size (v, v, n) where v is the number of nodes, and n is the number of subjects.
    :type x: Numpy Array.
    :param y: behavioral data of size (n,).
    :type y: Numpy Array.
    :param k: number of folds.
    :type k: integer.
    :param out_path: output directory.
    :type out_path: string.
    :param p_thresh: p threshold for determining significant edges.
    :type p_thresh: float.
    :param iter: iteration number, 1-indexing.
    :type iter: integer.
    :param zscore: whether to z-score edge strength in the training set.
    :type zscore: bool
    :param mode: what function to use when determining edges. 'linear' or 'ridge'.
    :type mode: string
    :return: None
    :rtype: None
    """
    print("---------------------------------------")
    print("Iteration #{}".format(iter))
    print("---------------------------------------")
    outputs = {}
    outputs['y_pred_pos'], outputs['y_pred_neg'], outputs['y_pred_both'], outputs['fit_p'], outputs['fit_n'], outputs[
        'fit_b'], outputs['edges_p'], outputs['edges_n'] = kfold_cpm(x, y, k, p_thresh, zscore, mode)
    save_run_outputs(out_path, iter, outputs, y, mode)
    return None

