import numpy as np
import pandas as pd
import sys
import scipy.io
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler


def save_run_outputs_subsample_nogrid(out_path, iter, outputs, y):
    """
    Save run outputs for subsample CPM
    :param out_path: output path
    :type out_path: string
    :param iter: iteration number
    :type iter: int
    :param outputs: outputs of kfold_cpm_subsample
    :type outputs: dict
    :param y: actual target behavioral data
    :type y: numpy array (n,)
    :return: None
    :rtype: None
    """
    for fold, (network_p, network_n) in enumerate(zip(outputs['edges_p'], outputs['edges_n'])):
        np.savetxt('{}/positive_network_from_training_iter{}_fold_{}.txt'.format(out_path, iter, fold + 1), network_p, fmt='%d')
        np.savetxt('{}/negative_network_from_training_iter{}_fold_{}.txt'.format(out_path, iter, fold + 1), network_n, fmt='%d')

    df_y_predict = pd.DataFrame(columns=['y_pred_both', 'y_actual'])
    df_y_predict['y_pred_both'] = outputs['y_pred_both'].flatten()
    df_y_predict['y_actual'] = y
    df_y_predict.to_csv('{}/y_prediction_iter{}.csv'.format(out_path, iter))

    df_fit = pd.DataFrame(columns=['both_m', 'both_b'],
                          index=['fold {}'.format(x + 1) for x in range(len(outputs['fit_b']))])
    df_fit['both_m'] = [fit.coef_[0][0] for fit in outputs['fit_b']]
    df_fit['both_b'] = [fit.intercept_[0] for fit in outputs['fit_b']]
    df_fit.to_csv('{}/fit_parameters_iter{}.csv'.format(out_path, iter))
    return None


def save_run_outputs_subsample(out_path, iter, outputs, y):
    """
    Save run outputs for subsample CPM
    :param out_path: output path
    :type out_path: string
    :param iter: iteration number
    :type iter: int
    :param outputs: outputs of kfold_cpm_subsample
    :type outputs: dict
    :param y: actual target behavioral data
    :type y: numpy array (n,)
    :return: None
    :rtype: None
    """
    for fold, (network_p, network_n) in enumerate(zip(outputs['edges_p'], outputs['edges_n'])):
        np.savetxt('{}/positive_network_from_training_iter{}_fold_{}.txt'.format(out_path, iter, fold + 1), network_p, fmt='%d')
        np.savetxt('{}/negative_network_from_training_iter{}_fold_{}.txt'.format(out_path, iter, fold + 1), network_n, fmt='%d')

    df_y_predict = pd.DataFrame(columns=['y_pred_both', 'y_actual'])
    df_y_predict['y_pred_both'] = outputs['y_pred_both'].flatten()
    df_y_predict['y_actual'] = y
    df_y_predict.to_csv('{}/y_prediction_iter{}.csv'.format(out_path, iter))

    df_fit = pd.DataFrame(columns=['both_m', 'both_b'],
                          index=['fold {}'.format(x + 1) for x in range(len(outputs['fit_b']))])
    df_fit['both_m'] = [fit.coef_[0][0] for fit in outputs['fit_b']]
    df_fit['both_b'] = [fit.intercept_[0] for fit in outputs['fit_b']]
    df_fit.to_csv('{}/fit_parameters_iter{}.csv'.format(out_path, iter))

    for fold, dict_best_params in enumerate(outputs['best_params']):
        df_params = pd.DataFrame(columns=list(dict_best_params.keys()), index=['fold {}'.format(fold+1)])
        for param in list(dict_best_params.keys()):
            df_params[param] = dict_best_params[param]
        df_params.to_csv('{}/best_params_iter{}_fold_{}.csv'.format(out_path, iter, fold+1))
    return None


def y_transform(y, y_norm='id'):
    """
    normalize all behavioral data
    :param y: list of all behavioral data, (n_subj,)
    :type y: numpy array
    :param y_norm: normalization method
    :type y_norm: 'id', 'yj', or 'norm'
    :return:
        yn: normalized list of all behavioral data, (n_subj,)
        transformer: trained transformer
    :rtype: (list of floats, sklearn object)
    """
    y = y.reshape(-1, 1)
    if y_norm == 'yj':
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        transformer.fit(y)
        yn = transformer.transform(y)
    elif y_norm == 'id':
        transformer = FunctionTransformer()  # identity function
        transformer.fit(y)
        yn = transformer.transform(y)
    elif y_norm == 'norm':
        transformer = StandardScaler()  # identity function
        transformer.fit(y)
        yn = transformer.transform(y)
    else:
        print("WARNING: undefined y_norm {}. Use identity function instead.".format(y_norm))
        transformer = FunctionTransformer()  # identity function
        transformer.fit(y)
        yn = transformer.transform(y)

    yn = yn.reshape(-1, )
    return yn, transformer


def save_matlab_mat(path, matname, x, y, lst_subj):
    """
    save mdict to matlab .mat file.
    :param path: path to output file.
    :type path: string.
    :param matname: name of the output .mat file.
    :type matname: string.
    :param x: stacked feature matrix of size (v, v, n) where v is the number of nodes, and n is the number of subjects.
    :type x: Numpy Array.
    :param y: true behavioral data of size (n,).
    :type y: Numpy Array.
    :param lst_subj: list of subject keys.
    :type lst_subj: list of strings.
    :return: None
    :rtype: None
    """
    mdict = {"x": x, "y": y, "subjectkey": lst_subj}
    scipy.io.savemat("{}/{}".format(path, matname), mdict)
    return None


def read_matlab_mat(path, matname):
    """
    Read matlab .mat files and return x, y
    :param path: path to file
    :type path: string
    :param matname: name of the .mat file
    :type matname: string
    :return:
        x: stacked feature matrix of size (v, v, n) where v is the number of nodes, and n is the number of subjects.
        y: true behavioral data of size (n,).
        lst_subjectkey: list of subject keys.
    :rtype: (NumPy Array, NumPy Array, list of strings)
    """
    mdict = scipy.io.loadmat("{}/{}".format(path, matname))
    x = mdict['x']
    y = mdict['y'][0]
    lst_subjectkey = mdict['subjectkey']
    return x, y, lst_subjectkey


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check symmetry of input matrix.
    :param a: input matrix.
    :type a: NumPy 2D Array.
    :param rtol: relative tolerance.
    :type rtol: float.
    :param atol: absolute tolerance.
    :type atol: float.
    :return: True or False
    :rtype: Boolean
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol, equal_nan=True)


def generate_file_list(path, lst_subj, num_roi, num_contrasts, t):
    """
    Generate list of files, where each file contains a single subject connectivity matrix.
    The files should have been generated by analysis_ABCD/make_coactivation_matrix.ipynb.
    :param path: path to where correlation matrices are saved.
    :type path: string.
    :param lst_subj: list of ABCD subjectkey.
    :type lst_subj: list of strings.
    :param num_roi: number of ROIs in the coactivation matrices
    :type num_roi: int
    :param num_contrasts: number of contrasts used to create coactivation matrices
    :type num_contrasts: int
    :param t: time point (bsl or y2).
    :type t: string.
    :return: list of files.
    :rtype: list of strings.
    """
    fn_list = []
    for subj in lst_subj:
        fn_list.append('{}/{}_{}ROI_{}contrasts_corr_matrix_{}.txt'.format(path, subj, num_roi, num_contrasts, t))
    return fn_list


def read_mats(fn_list):
    """
    Read list of single-subject connectivity matrix and return stacked matrices.
    :param fn_list: list of files, where each contains a single subject connectivity matrix.
    :type fn_list: list of strings.
    :return: stacked matrix of size (v, v, n), where v is the number of nodes, and n is the number of subjects.
    :rtype: Numpy Array.
    """
    fns = [pd.read_csv(fn, sep=' ', header=None) for fn in fn_list]
    if sum([df.isnull().values.any() for df in fns]) != 0:  # check for NaN
        sys.exit("ERROR: there are NaNs in the correlation matrices! Please check your data.")
    fns = [df.values for df in fns]
    fn_mats = np.stack(fns, axis=2)  # join the (v, v) arrays on a new axis (the third axis)
    return fn_mats


def return_estimator_coef(est, mode):
    """
    Return coefficients of an estimator.
    :param est: trained estimator
    :type est: a sklearn estimator, or np.nan
    :param mode: what function to use when determining edges. 'linear' or 'ridge'.
    :type mode: string
    :return: all related parameters to the estimator
    :rtype: floats
    """
    if mode == 'linear' or mode == 'logistic':
        if isinstance(est, float):  # if the estimator is np.nan
            return [np.nan, np.nan]
        else:
            return [est.coef_[0][0], est.intercept_[0]]
    elif mode == 'ridge':
        if isinstance(est, float):
            return [np.nan, np.nan, np.nan]
        else:
            return [est.best_estimator_.coef_[0][0], est.best_estimator_.intercept_[0], est.best_params_['alpha']]
    else:
        print("ERROR: mode {} not implemented!".format(mode))
        quit()


def save_run_outputs(out_path, iter, outputs, y_run, mode='linear'):
    """
    Save k-fold CPM outputs.
    :param out_path: output directory.
    :type out_path: string.
    :param iter: iteration number.
    :type iter: integer.
    :param outputs: outputs from kfold_cpm.
    :type outputs: dict.
    :param y_run: input behav data for kfold_cpm.
    :type y_run: NumPy Array.
    :param mode: what function to use when determining edges. 'linear' or 'ridge'.
    :type mode: string
    :return: None
    :rtype: None
    """
    for fold, (network_p, network_n) in enumerate(zip(outputs['edges_p'], outputs['edges_n'])):
        np.savetxt('{}/positive_network_from_training_iter{}_fold_{}.txt'.format(out_path, iter, fold + 1), network_p, fmt='%d')
        np.savetxt('{}/negative_network_from_training_iter{}_fold_{}.txt'.format(out_path, iter, fold + 1), network_n, fmt='%d')

    df_y_predict = pd.DataFrame(columns=['y_pred_pos', 'y_pred_neg', 'y_pred_both', 'y_actual'])
    df_y_predict['y_pred_pos'] = outputs['y_pred_pos']
    df_y_predict['y_pred_neg'] = outputs['y_pred_neg']
    df_y_predict['y_pred_both'] = outputs['y_pred_both']
    df_y_predict['y_actual'] = y_run
    df_y_predict.to_csv('{}/y_prediction_iter{}.csv'.format(out_path, iter))

    df_fit = pd.DataFrame(columns=['pos_m', 'pos_b', 'neg_m', 'neg_b', 'both_m', 'both_b'], index=['fold {}'.format(x + 1) for x in range(len(outputs['fit_p']))])
    df_fit['pos_m'] = [return_estimator_coef(fit, mode)[0] for fit in outputs['fit_p']]
    df_fit['pos_b'] = [return_estimator_coef(fit, mode)[1] for fit in outputs['fit_p']]
    df_fit['neg_m'] = [return_estimator_coef(fit, mode)[0] for fit in outputs['fit_n']]
    df_fit['neg_b'] = [return_estimator_coef(fit, mode)[1] for fit in outputs['fit_n']]
    df_fit['both_m'] = [return_estimator_coef(fit, mode)[0] for fit in outputs['fit_b']]
    df_fit['both_b'] = [return_estimator_coef(fit, mode)[1] for fit in outputs['fit_b']]
    if mode == 'ridge':
        df_fit['pos_alpha'] = [return_estimator_coef(fit, mode)[2] for fit in outputs['fit_p']]
        df_fit['neg_alpha'] = [return_estimator_coef(fit, mode)[2] for fit in outputs['fit_n']]
        df_fit['both_alpha'] = [return_estimator_coef(fit, mode)[2] for fit in outputs['fit_b']]
    df_fit.to_csv('{}/fit_parameters_iter{}.csv'.format(out_path, iter))

    return None
