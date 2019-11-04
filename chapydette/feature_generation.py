# coding=utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.spatial.distance
import sklearn.cluster
import sklearn.decomposition
import sklearn.mixture
import sklearn.preprocessing
import torch

try:
    import faiss
    print('Using Faiss')
    USE_FAISS = 1
except:
    USE_FAISS = 0
    print('Not using faiss')

try:
    import pomegranate
    print('Using pomegranate')
    USE_POMEGRANATE = 1
except:
    USE_POMEGRANATE = 0
    print('Not using pomegranate')

try:
    from yael import yael, ynumpy
    print('Using Yael')
    USE_YAEL = 1
except:
    print('Not using Yael')
    USE_YAEL = 0

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


def nystroem_features(data_features, nclusters, window_length=1, data_codebook=None, times=None, bandwidth=None,
                      window_overlap='default', standardize=True, scaler=None, do_pca=True, pca_frac=0.95,
                      pca_result=None, centroids=None, kmeans_iters=100, ngpu=1, njobs=1, seed=None):
    """
    Generate features based on the Nystroem method for the Gaussian RBF kernel for the input data data_features.
    Reference:

    - Williams, C. K., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. In Advances in Neural Information Processing Systems (pp. 682-688).

    :param data_features: Data that features will be generated for. Observations are the rows.
    :type data_features: numpy.ndarray
    :param nclusters: Number of clusters to use in codebook generation.
    :type nclusters: int
    :param window_length: Sliding window length (in terms of time- see below).
    :type window_length: float
    :param data_codebook: Data to use for codebook generation. If None, then data_features is used.
    :type data_codebook: numpy.ndarray
    :param times: Times corresponding to each observation. If None, the indices of each observation will be used.
    :type times: numpy.array
    :param bandwidth: Bandwidth to use for the RBF kernel in the Nystroem computation. If None, the median pairwise
                      distance rule of thumb is used.
    :type bandwidth: float
    :param window_overlap: Amount of overlap between sliding windows, in terms of time. Default: 0.2*window_length.
    :type window_overlap: float
    :param standardize: Whether to standardize the input data
    :type standardize: bool
    :param scaler: Scikit-learn scaling object from prior scaling.
    :param do_pca: Whether or not to perform PCA on the data.
    :type do_pca: bool
    :param pca_frac: Percentage of variance to be retained after running PCA.
    :type pca_frac: float
    :param pca_result: Faiss or scikit-learn pca object from prior run of PCA.
    :param centroids: Centroids from previous run of codebook generation.
    :type centroids: numpy.ndarray
    :param kmeans_iters: Number of iterations of k-means to perform.
    :type kmeans_iters: int
    :param ngpu: Number of GPUs to use. Unused if no GPUs are available.
    :type ngpu: int
    :param njobs: Number of parallel jobs to run.
    :type njobs: int
    :param seed: Seed for reproducibility.
    :type seed: int
    :return: (tuple): tuple containing:

            * features (numpy.ndarray): Bag-of-features features.
            * mirrored_times[start_idxs.astype('int')] (numpy.array): Start time of each interval.
            * mirrored_times[end_idxs.astype('int')] (numpy.array): End time of each interval.
            * scaler: Scikit-learn scaler object.
            * pca: Faiss or scikit-learn pca object.
            * centroids (numpy.ndarray): Codebook (centroids from k-means)
    """
    data_codebook, mirrored_features, mirrored_times, start_idxs, end_idxs, scaler, pca = prep_features(data_codebook,
                                                                                                        data_features,
                                                                                                        times,
                                                                                                        standardize,
                                                                                                        scaler,
                                                                                                        do_pca,
                                                                                                        pca_frac,
                                                                                                        pca_result,
                                                                                                        window_length,
                                                                                                        window_overlap)
    if centroids is None:
        centroids = kmeans(data_codebook, nclusters, kmeans_iters, ngpu, njobs, seed)
    else:
        if np.size(centroids, 1) != np.size(data_features, 1):
            raise ValueError('Centroids must have same size of second axis (axis 1) as data_features')

    features = generate_nystroem(mirrored_features, centroids, start_idxs, end_idxs, bandwidth)
    features = features.cpu().double().numpy()

    print('Done generating features.')
    return features, mirrored_times[start_idxs.astype('int')], mirrored_times[end_idxs.astype('int')], scaler, pca, centroids



def bag_of_features(data_features, nclusters, window_length, data_codebook=None, times=None, window_overlap='default',
                    standardize=True, scaler=None, do_pca=True, pca_frac=0.95, pca_result=None, centroids=None,
                    kmeans_iters=100, ngpu=1, njobs=1, seed=None):
    """
    Generate bag-of-features features for the input data data_features.

    :param data_features: Data that features will be generated for. Observations are the rows.
    :type data_features: numpy.ndarray
    :param nclusters: Number of clusters to use in codebook generation.
    :type nclusters: int
    :param window_length: Sliding window length (in terms of time- see below).
    :type window_length: float
    :param data_codebook: Data to use for codebook generation. If None, then data_features is used.
    :type data_codebook: numpy.ndarray
    :param times: Times corresponding to each observation. If None, the indices of each observation will be used.
    :type times: numpy.array
    :param window_overlap: Amount of overlap between sliding windows, in terms of time. Default: 0.2*window_length.
    :type window_overlap: float
    :param standardize: Whether to standardize the input data
    :type standardize: bool
    :param scaler: Scikit-learn scaling object from prior scaling.
    :param do_pca: Whether or not to perform PCA on the data.
    :type do_pca: bool
    :param pca_frac: Percentage of variance to be retained after running PCA.
    :type pca_frac: float
    :param pca_result: Faiss or scikit-learn pca object from prior run of PCA.
    :param centroids: Centroids from previous run of codebook generation.
    :type centroids: numpy.ndarray
    :param kmeans_iters: Number of iterations of k-means to perform.
    :type kmeans_iters: int
    :param ngpu: Number of GPUs to use. Unused if no GPUs are available.
    :type ngpu: int
    :param njobs: Number of parallel jobs to run.
    :type njobs: int
    :param seed: Seed for reproducibility.
    :type seed: int
    :return: (tuple): tuple containing:

            * features (numpy.ndarray): Bag-of-features features.
            * mirrored_times[start_idxs.astype('int')] (numpy.array): Start time of each interval.
            * mirrored_times[end_idxs.astype('int')] (numpy.array): End time of each interval.
            * scaler: Scikit-learn scaler object.
            * pca: Faiss or scikit-learn pca object.
            * centroids (numpy.ndarray): Codebook (centroids from k-means)
    """
    data_codebook, mirrored_features, mirrored_times, start_idxs, end_idxs, scaler, pca = prep_features(data_codebook,
                                                                                                        data_features,
                                                                                                        times,
                                                                                                        standardize,
                                                                                                        scaler,
                                                                                                        do_pca,
                                                                                                        pca_frac,
                                                                                                        pca_result,
                                                                                                        window_length,
                                                                                                        window_overlap)
    if centroids is None:
        centroids = kmeans(data_codebook, nclusters, kmeans_iters, ngpu, njobs, seed)
    else:
        if np.size(centroids, 1) != np.size(data_features, 1):
            raise ValueError('Centroids must have same size of second axis (axis 1) as data_features')

    features = generate_bof(mirrored_features, centroids, start_idxs, end_idxs)

    print('Done generating features.')
    return features, mirrored_times[start_idxs.astype('int')], mirrored_times[end_idxs.astype('int')], scaler, pca, centroids


def vlad(data_features, nclusters, window_length, data_codebook=None, times=None, window_overlap='default',
         standardize=True, scaler=None, do_pca=True, pca_frac=0.95, pca_result=None, centroids=None, kmeans_iters=1000,
         ngpu=1, njobs=1, seed=None):
    """
    Generate VLAD features for the input data data_features.

    Reference:

    - Jégou, H., Douze, M., Schmid, C., & Pérez, P. (2010, June). Aggregating local descriptors into a compact image representation. In Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on (pp. 3304-3311). IEEE.

    :param data_features: Data that features will be generated for. Observations are the rows.
    :type data_features: numpy.ndarray
    :param nclusters: Number of clusters to use in codebook generation.
    :type nclusters: int
    :param window_length: Sliding window length (in terms of time- see below).
    :type window_length: float
    :param data_codebook: Data to use for codebook generation. If None, then data_features is used.
    :type data_codebook: numpy.ndarray
    :param times: Times corresponding to each observation. If None, the indices of each observation will be used.
    :type times: numpy.array
    :param window_overlap: Amount of overlap between sliding windows, in terms of time. Default: 0.2*window_length.
    :type window_overlap: float
    :param standardize: Whether to standardize the input data
    :type standardize: bool
    :param scaler: Scikit-learn scaling object from prior scaling.
    :param do_pca: Whether or not to perform PCA on the data.
    :type do_pca: bool
    :param pca_frac: Percentage of variance to be retained after running PCA.
    :type pca_frac: float
    :param pca_result: Faiss or scikit-learn pca object from prior run of PCA.
    :param centroids: Centroids from previous run of codebook generation.
    :type centroids: numpy.ndarray
    :param kmeans_iters: Number of iterations of k-means to perform.
    :type kmeans_iters: int
    :param ngpu: Number of GPUs to use. Unused if no GPUs are available.
    :type ngpu: int
    :param njobs: Number of parallel jobs to run.
    :type njobs: int
    :param seed: Seed for reproducibility.
    :type seed: int
    :return: (tuple): tuple containing:

            * features (numpy.ndarray): VLAD features.
            * mirrored_times[start_idxs.astype('int')] (numpy.array): Start time of each interval.
            * mirrored_times[end_idxs.astype('int')] (numpy.array): End time of each interval.
            * scaler: Scikit-learn scaler object.
            * pca: Faiss or scikit-learn pca object.
            * centroids (numpy.ndarray): Codebook (centroids from k-means)
    """
    data_codebook, mirrored_features, mirrored_times, start_idxs, end_idxs, scaler, pca = prep_features(data_codebook,
                                                                                                        data_features,
                                                                                                        times,
                                                                                                        standardize,
                                                                                                        scaler,
                                                                                                        do_pca,
                                                                                                        pca_frac,
                                                                                                        pca_result,
                                                                                                        window_length,
                                                                                                        window_overlap)
    if centroids is None:
        centroids = kmeans(data_codebook, nclusters, kmeans_iters, ngpu, njobs, seed)
    else:
        if np.size(centroids, 1) != np.size(data_features, 1):
            raise ValueError('Centroids must have same size of the second axis (axis 1) as data_features')

    features = generate_vlad(mirrored_features, centroids, start_idxs, end_idxs)

    print('Done generating features.')
    return features, mirrored_times[start_idxs.astype('int')], mirrored_times[end_idxs.astype('int')], scaler, pca, centroids


def fisher_vectors(data_features, nclusters, window_length, data_codebook=None, times=None, window_overlap='default',
                   standardize=True, scaler=None, do_pca=True, pca_frac=0.95, pca_result=None, gmm_result=None,
                   gmm_iters=1000, ngpu=1, njobs=1, seed=None):
    """
    Generate Fisher vector features for the input data data_features.

    References:

    * Jaakkola, T., & Haussler, D. (1999). Exploiting generative models in discriminative classifiers. In Advances in neural information processing systems (pp. 487-493).
    * Perronnin, F., & Dance, C. (2007, June). Fisher kernels on visual vocabularies for image categorization. In Computer Vision and Pattern Recognition, 2007. CVPR'07. IEEE Conference on (pp. 1-8). IEEE.

    :param data_features: Data that features will be generated for. Observations are the rows.
    :type data_features: numpy.ndarray
    :param nclusters: Number of components to use when fitting the Gaussian mixture model.
    :type nclusters: int
    :param window_length: Sliding window length (in terms of time- see below).
    :type window_length: float
    :param data_codebook: Data to use for fitting the Gaussian mixture model. If None, then data_features is used.
    :type data_codebook: numpy.ndarray
    :param times: Times corresponding to each observation. If None, the indices of each observation will be used.
    :type times: numpy.array
    :param window_overlap: Amount of overlap between sliding windows, in terms of time. Default: 0.2*window_length.
    :type window_overlap: float
    :param standardize: Whether to standardize the input data
    :type standardize: bool
    :param scaler: Scikit-learn scaling object from prior scaling.
    :param do_pca: Whether or not to perform PCA on the data.
    :type do_pca: bool
    :param pca_frac: Percentage of variance to be retained after running PCA.
    :type pca_frac: float
    :param pca_result: Faiss or scikit-learn pca object from prior run of PCA.
    :param gmm_result: GMM object from previous run.
    :param gmm_iters: Maximum number of iterations to perform when fitting the GMM.
    :type gmm_iters: int
    :param ngpu: Number of GPUs to use. Unused if no GPUs are available.
    :type ngpu: int
    :param njobs: Number of parallel jobs to run.
    :type njobs: int
    :param seed: Seed for reproducibility.
    :type seed: int
    :return: (tuple): tuple containing:

            * features (numpy.ndarray): Fisher vector features.
            * mirrored_times[start_idxs.astype('int')] (numpy.array): Start time of each interval.
            * mirrored_times[end_idxs.astype('int')] (numpy.array): End time of each interval.
            * scaler: Scikit-learn scaler object.
            * pca: Faiss or scikit-learn pca object.
            * gmm_result: GMM object from faiss or scikit-learn
    """
    data_codebook, mirrored_features, mirrored_times, start_idxs, end_idxs, scaler, pca = prep_features(data_codebook,
                                                                                                        data_features,
                                                                                                        times,
                                                                                                        standardize,
                                                                                                        scaler,
                                                                                                        do_pca,
                                                                                                        pca_frac,
                                                                                                        pca_result,
                                                                                                        window_length,
                                                                                                        window_overlap)
    if gmm_result is None:
        ws, mus, sigmas, gmm_result = gmm(mirrored_features, nclusters, num_iters=gmm_iters, njobs=njobs, nredo=10, seed=seed)
    else:
        if hasattr(gmm_result, 'weights_'):
            ws, mus, sigmas = gmm_result.weights_, gmm_result.means_, gmm_result.covariances_
        else:
            try:
                ws, mus, sigmas = gmm_result[0], gmm_result[1], gmm_result[2]
            except:
                raise NotImplementedError('Unrecognized type for gmm_result')
        nclusters = len(ws)
        if np.size(mus, 1) != np.size(data_features, 1):
            raise ValueError('Means and covariances must have same size of the second axis (axis 1) as data_features')

    features = generate_fisher(mirrored_features, (ws, mus, sigmas), gmm_result, nclusters, start_idxs, end_idxs)

    print('Done generating features.')
    return features, mirrored_times[start_idxs.astype('int')], mirrored_times[end_idxs.astype('int')], scaler, pca, gmm_result


def prep_features(data_codebook, data_features, times, standardize, scaler, do_pca, pca_frac, pca_result, window_length,
                  window_overlap):
    """
    Prepare the features by standardizing them, running PCA, mirroring the features, and finding the start and end
    indices for the sliding windows.

    :param data_codebook: Data to use for codebook generation. If None, then data_features is used.
    :param data_features: Data that features will be generated for. Observations are the rows.
    :param times: Times corresponding to each observation. If None, the indices of each observation will be used.
    :param standardize: Whether to standardize the input data
    :param scaler: Scikit-learn scaling object from prior scaling.
    :param do_pca: Whether or not to perform PCA on the data.
    :param pca_frac: Percentage of variance to be retained after running PCA.
    :param pca_result: Faiss or scikit-learn pca object from prior run of PCA.
    :param window_length: Sliding window length (in terms of time).
    :param window_overlap: Amount of overlap between sliding windows, in terms of time. Default: 0.2*window_length.
    :return: data_codebook: Data to use for codebook generation.
    :return: mirrored_features: data_features with the observations at the beginning and end mirrored.
    :return: mirrored_times: Times corresponding to the mirrored features.
    :return: start_idxs: Indices of the start of each sliding window.
    :return: end_idxs: Indices of the end of each sliding window.
    :return: scaler: Scikit-learn scaler object.
    :return: pca: Faiss or scikit-learn pca object.
    """
    if times is None:
        times = range(np.size(data_features, 0))
    times = np.array(times)
    assert len(times) == np.size(data_features, 0), 'The time axis for the input features must be axis 0 and must be' \
                                                    ' the same length as times'
    if window_overlap == 'default':
        window_overlap = 0.2*window_length

    if pca_result and not scaler:
        raise ValueError('If you provide an input for pca_result you must also provide scaler.')

    if data_codebook is None:
        data_codebook = data_features
        reuse_data = True
    else:
        reuse_data = False
    if (not scaler) and standardize:
        scaler = sklearn.preprocessing.StandardScaler().fit(data_codebook)
        scaled_features = scaler.transform(data_codebook)
    elif standardize:
        scaled_features = scaler.transform(data_codebook)
    else:
        scaled_features = data_codebook
    if do_pca:
        data_codebook, pca = run_pca(scaled_features, pca_frac)
    elif pca_result:
        pca = pca_result
        data_codebook = pca.transform(data_codebook)
        do_pca = True
    else:
        pca = None
    if do_pca and not reuse_data:
        if USE_FAISS:
            scaled_features = np.ascontiguousarray(scaler.transform(data_features)).astype('float32')
        data_features = pca.transform(scaled_features)
    elif standardize and not reuse_data:
        data_features = scaler.transform(data_features)
    else:
        data_features = data_codebook
    mirrored_features, mirrored_times = mirror_features(data_features, times, window_length, window_overlap)
    start_idxs, end_idxs = get_window_start_end_idxs(times, mirrored_times, window_length, window_overlap)

    return data_codebook, mirrored_features, mirrored_times, start_idxs, end_idxs, scaler, pca


def run_pca(scaled_features, pca_frac):
    """
    Run PCA on scaled features using either faiss (if available) or scikit-learn (otherwise).

    :param scaled_features: Scaled features to run PCA on
    :param pca_frac: Percent of the variance to retain
    :return: pca_features: scaled_features projected to the lower dimensional space
    :return: pca: PCA object from faiss or scikit-learn
    """
    print('Running PCA...')
    if USE_FAISS:
        d = np.size(scaled_features, 1)
        pca = faiss.PCAMatrix(d, d)
        x = np.ascontiguousarray(scaled_features).astype('float32')
        pca.train(x)
        assert pca.is_trained
        eigs = faiss.vector_float_to_array(pca.eigenvalues)
        explained_var = np.cumsum(eigs)/sum(eigs)
        num_retain = np.where(explained_var >= pca_frac)[0][0]
        pca_features = pca.apply_py(x)
        pca_features = pca_features[:, 0:(num_retain+1)]
        pca.transform = pca.apply_py
    else:
        pca = sklearn.decomposition.PCA(pca_frac, svd_solver='full')
        pca.fit(scaled_features)
        pca_features = pca.transform(scaled_features)

    return pca_features, pca


def mirror_features(features, times, window_length, window_overlap):
    """
    Mirror the features so that the first interval is centered at zero and so the last observations are also mirrored.

    :param features: Features to be mirrored.
    :param times: Times corresponding to the above features.
    :param window_length: Sliding window length (in terms of time).
    :param window_overlap: Amount of overlap between sliding windows, in terms of time.
    :return: mirrored_features: Features that have been mirrored
    :return: mirrored_times: Times corresponding to the mirrored features.
    """
    if window_overlap != 0:
        mirror_start_idxs = sorted(np.where(times <= window_length / 2)[0], reverse=True)[:-1]
        mirror_end_idxs = sorted(np.where(times > times[-1] - window_length)[0], reverse=True)[1:]
        mirrored_features = np.vstack((features[mirror_start_idxs, :], features, features[mirror_end_idxs, :]))
        mirrored_times = np.hstack((-1*times[mirror_start_idxs], times, 2*times[-1] - times[mirror_end_idxs]))
    else:
        mirrored_features = features
        mirrored_times = times

    return mirrored_features, mirrored_times


def get_window_start_end_idxs(times, mirrored_times, window_length, window_overlap, eps=1e-10):
    """
    Find the start and end indices of each sliding window.

    :param times: Times corresponding to each observation.
    :param mirrored_times: Times corresponding to each mirrored observation.
    :param window_length: Sliding window length (in terms of time).
    :param window_overlap: Amount of overlap between sliding windows, in terms of time.
    :return: start_idxs: Start indices of each sliding window
    :return: end_idxs: End indices of each sliding window
    """
    start_times = [mirrored_times[0]]
    found_endpoint = 0
    current_time = mirrored_times[0]
    while not found_endpoint:
        current_time += window_length - window_overlap
        start_times.append(current_time)
        if current_time + (window_length-1) >= times[-1]:
            found_endpoint = 1
    start_times = np.array(start_times)
    end_times = start_times + (window_length-eps)

    start_idxs = np.zeros(len(start_times))
    end_idxs = np.zeros(len(end_times))
    for i in range(len(start_times)):
        idxs = sorted(np.where((start_times[i] <= mirrored_times) & (mirrored_times < end_times[i]))[0])
        if len(idxs) != 0:
            start_idxs[i] = idxs[0]
            end_idxs[i] = idxs[-1]
        else:
            start_idxs[i] = -1
            end_idxs[i] = -1
    # Remove places where there are no obs
    null_idxs = np.where((start_idxs == end_idxs) & (start_idxs == -1))[0]
    start_idxs = np.delete(start_idxs, null_idxs)
    end_idxs = np.delete(end_idxs, null_idxs)

    return start_idxs, end_idxs


def kmeans(features, nclusters, num_iters, ngpu, njobs, seed):
    """
    Run k-means on features, generating nclusters clusters. It will use, in order of preference, Faiss, pomegranate, or
    scikit-learn.

    :param features: Features to cluster.
    :param nclusters: Number of clusters to generate.
    :param num_iters: Maximum number of iterations to perform.
    :param ngpu: Number of GPUs to use (if GPUs are available).
    :param njobs: Number of threads to use.
    :param seed: Seed for reproducibility.
    :return: centroids: The centroids found with k-means.
    """
    print('Running k-means...')
    if USE_FAISS:
        d = features.shape[1]
        pca_features = np.ascontiguousarray(features).astype('float32')

        clus = faiss.Clustering(d, nclusters)
        clus.verbose = True
        clus.niter = num_iters
        if seed is not None:
            clus.seed = seed

        # otherwise the kmeans implementation sub-samples the training set
        clus.max_points_per_centroid = 10000000

        if USE_GPU:
            res = [faiss.StandardGpuResources() for i in range(ngpu)]

            flat_config = []
            for i in range(ngpu):
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = i
                flat_config.append(cfg)

            if ngpu == 1:
                index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
            else:
                indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                           for i in range(ngpu)]
                index = faiss.IndexProxy()
                for sub_index in indexes:
                    index.addIndex(sub_index)
        else:
            index = faiss.IndexFlatL2(d)

        clus.train(pca_features, index)
        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = centroids.reshape(nclusters, d)

    elif USE_POMEGRANATE and seed is None:
        kmeans = pomegranate.kmeans.Kmeans(nclusters, init='kmeans++', n_init=10)
        kmeans.fit(features, max_iterations=num_iters, n_jobs=njobs)
        centroids = kmeans.centroids
    else:
        if USE_POMEGRANATE and seed is not None:
            print('Pomegranate does not currently support k-means with a seed. Switching to scikit-learn instead.')
        print('Using scikit-learn. This may be slow!')
        kmeans = sklearn.cluster.KMeans(n_clusters=nclusters, random_state=seed).fit(features)
        centroids = kmeans.cluster_centers_

    return centroids


def gmm(features, ncomponents, num_iters, njobs, nredo, seed):
    """
    Fit a Gaussian Mixture Model with nclusters using either Yael (if available) or scikit-learn (otherwise). The GMM
    assumes a diagonal covariance matrix.

    :param features: Features to use to fit the GMM.
    :param ncomponents: Number of components in the GMM.
    :param num_iters: Maximum number of iterations to perform when fitting the GMM.
    :param njobs: Number of threads to use.
    :param nredo: Number of initializations to perform.
    :param seed: Seed for reproducibility
    :return: weights: GMM estimated weights
    :return: mus: GMM estimated means
    :return: sigmas: Diagonals of GMM estimated covariance matrices
    :return: gmm: GMM object from faiss or scikit-learn
    """
    print('Running GMM...')
    # if USE_POMEGRANATE:
    #     gmm = pomegranate.GeneralMixtureModel.from_samples(pomegranate.MultivariateGaussianDistribution, n_components=nclusters, X=features, n_init=nredo)
    #     weights = np.exp(gmm.weights)
    #     mus = np.array([gmm.distributions[i].parameters[0] for i in range(len(gmm.distributions))])
    #     sigmas = np.array([np.diag(gmm.distributions[i].parameters[1]) for i in range(len(gmm.distributions))])
    if USE_YAEL:
        pca_features = np.ascontiguousarray(features).astype('float32')
        if seed is None:
            seed = 0
        gmm = ynumpy.gmm_learn(pca_features, ncomponents, nt=njobs, niter=num_iters, redo=nredo, seed=seed)
        weights, mus, sigmas = gmm[0], gmm[1], gmm[2]
    else:
        gmm = sklearn.mixture.GaussianMixture(n_components=ncomponents, covariance_type='diag', n_init=nredo, random_state=seed)
        gmm.fit(features)
        weights, mus, sigmas = gmm.weights_, gmm.means_, gmm.covariances_

    return weights, mus, sigmas, gmm


def find_closest_centroid(centroids, features):
    """
    Find the closest centroid to each feature in terms of l2 distance.

    :param centroids: Centroids from codebook generation.
    :param features: Features for which you want to compute the nearest centroid.
    :return: idxs: Indices of the nearest centroid to each observation in features.
    """
    if USE_FAISS:
        centroids = np.ascontiguousarray(centroids).astype('float32')
        features = np.ascontiguousarray(features).astype('float32')
        nq, d = centroids.shape
        if USE_GPU:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0

            index = faiss.GpuIndexFlatL2(res, d, flat_config)
        else:
            index = faiss.IndexFlatL2(d)

        # add vectors to index
        index.add(centroids)
        # find nearest neighbors
        D, idxs = index.search(features, 1)
        idxs = idxs.flatten()
    else:
        idxs = np.argmin(scipy.spatial.distance.cdist(centroids, features), axis=0)

    return idxs


def rbf_kernel(x, y, bandwidth=None):
    """
    Compute the value of the rbf kernel between every row of x and every row of y

    :param x: Data with dimensions number of examples x number of features.
    :param y: Data with dimensions number of examples x number of features.
    :param bandwidth: Bandwidth for the RBF kernel. If None, the median pairwise distance rule of thumb is used.
    :return: gram: The gram matrix between x and y.
    :return: bandwidth: The bandwidth used for the kernel.
    """
    norm = squared_l2_norm(x, y)
    if bandwidth is None:
        bandwidth = torch.median(torch.sqrt(torch.max(norm, torch.DoubleTensor([1e-6]).to(DEVICE))))
        print('Using rule of thumb bandwidth:', bandwidth.item())
    gram = torch.exp(-1 / (2 * bandwidth ** 2) * norm)

    return gram, bandwidth


def squared_l2_norm(x, y):
    """
    Computed the squared l2 distance between each row of x and each row of y

    :param x: Data with dimensions number of examples x number of features.
    :param y: Data with dimensions number of examples x number of features.
    :return: Squared l2 distance between each row of x and each row of y
    """
    nx = x.shape[0]
    ny = y.shape[0]

    norm_x = torch.sum(x ** 2, 1).unsqueeze(0)
    norm_y = torch.sum(y ** 2, 1).unsqueeze(0)

    ones_x = torch.ones(nx, 1).double().to(DEVICE)
    ones_y = torch.ones(ny, 1).double().to(DEVICE)

    a = torch.mm(ones_y, norm_x)  # ny*nx
    b = torch.mm(x, y.t())  # nx*ny
    c = torch.mm(ones_x, norm_y)  # nx*ny

    return a.t() - 2 * b + c


def matrix_inv_sqrt(matrix, eps=1e-12, reg=0.001):
    """
    Given an input matrix, compute the matrix inverse square root (matrix)^{-1/2}

    :param matrix: Input matrix to compute the inverse square root of.
    :param eps: Threshold for the minum singular value
    :param reg: Regularization, in case the input matrix is poorly conditioned
    :return: normalization: The matrix inverse square root (matrix)^{-1/2} of the input matrix
    """
    U, S, V = torch.svd(matrix.to(DEVICE))
    eps = torch.Tensor([eps]).double().to(DEVICE)
    sqrt_S = torch.sqrt(torch.max(S, eps) + reg)
    normalization = torch.mm(torch.div(U, sqrt_S), V.t())

    return normalization


def generate_nystroem(features, centroids, start_idxs, end_idxs, bandwidth, return_last_features=False):
    """
    Generate features for each time period based on the Nystroem method. This computes the Nystroem features for every
    data point and then averages them within each input interval.

    :param features: Features to use in the Nystroem approximation
    :param centroids: Centroids used to determine the subspace to project onto in the Nystroem method
    :param start_idxs: Starting point of each time interval
    :param end_idxs: Ending point of each time interval
    :param bandwidth: Bandwidth for the RBF kernel
    :param return_last_features: Whether to return the non-averaged features for the last interval (for debugging
                                 purposes)
    :return: mean_nystroem_features: Averaged Nystroem features for each interval
    :return: features: The non-averaged features for the last interval
    """
    ndistns = len(start_idxs)
    mean_nystroem_features = torch.zeros((ndistns, centroids.shape[0]))
    features = torch.DoubleTensor(features).to(DEVICE)
    centroids = torch.DoubleTensor(centroids).to(DEVICE)
    kww, bandwidth = rbf_kernel(centroids, centroids, bandwidth=bandwidth)
    kww_inv_sqrt = matrix_inv_sqrt(kww)
    print('Generating features...')
    for i in range(ndistns):
        if i % 100 == 0:
            print('\r%0.2f' % (i*1.0 / ndistns * 100), '% done', end='')
        kwx, _ = rbf_kernel(centroids, features[int(start_idxs[i]):(int(end_idxs[i])+1)], bandwidth=bandwidth)
        nystroem_features = (kww_inv_sqrt.mm(kwx)).t()
        mean_nystroem_features[i] = nystroem_features.mean(0).cpu()
    print('\r100.00 % done')

    if not return_last_features:
        return mean_nystroem_features
    else:
        return mean_nystroem_features, nystroem_features


def generate_bof(mirrored_features, centroids, start_idxs, end_idxs):
    """
    Generate bag-of-features features for mirrored_features.

    :param mirrored_features: Features to compute bag-of-features features for.
    :param centroids: Codebook.
    :param start_idxs: Start indices of each sliding window.
    :param end_idxs: End indices of each sliding window.
    :return: bof: Bag-of-features features for mirrored_features.
    """
    print('Generating histograms...')
    closest_centroids = find_closest_centroid(centroids, mirrored_features)
    bof = np.zeros((len(start_idxs), np.size(centroids, 0)))
    for i in range(len(start_idxs)):
        if start_idxs[i] != -1:
            for closest_idx in closest_centroids[int(start_idxs[i]):int(end_idxs[i]) + 1]:
                bof[i, closest_idx] += 1

    denom = np.sum(bof, axis=1)
    denom[np.where(denom == 0)] = 1
    bof = bof/denom[:, np.newaxis]
    return bof


def generate_vlad(features, centroids, start_idxs, end_idxs):
    """
    Generate VLAD features for features.

    :param features: Features to generate VLAD features from.
    :param centroids: Codebook.
    :param start_idxs: Start indices of each sliding window.
    :param end_idxs: End indices of each sliding window.
    :return: vlad_features: VLAD features for features.
    """
    print('Generating VLAD features...')
    nintervals = len(start_idxs)
    ncentroids, d = centroids.shape
    vlad_features = np.zeros((nintervals, ncentroids * d))
    if not USE_YAEL or not hasattr(ynumpy, 'vlad'):
        closest_centroids = find_closest_centroid(centroids, features)
    for i in range(len(start_idxs)):
        X = features[int(start_idxs[i]):int(end_idxs[i]) + 1, :]
        if USE_YAEL and hasattr(ynumpy, 'vlad'):
            centroids = np.ascontiguousarray(centroids, dtype='float32')
            X = np.ascontiguousarray(X, dtype='float32')
            vlad_features[i, :] = ynumpy.vlad(centroids, X).flatten()
        else:
            for k in range(len(X)):
                assign = closest_centroids[int(k+start_idxs[i])]
                vlad_features[i, assign * d:(assign + 1) * d] += X[k, :] - centroids[assign, :]

    # Sign-square-root and normalize
    vlad_features = power_l2_normalize(vlad_features)

    return vlad_features


def generate_fisher(mirrored_features, gmm_results, gmm_object, ncomponents, start_idxs, end_idxs):
    """
    Generate Fisher vector features for mirrored_features.

    :param mirrored_features: Features to compute Fisher vector features for.
    :param gmm_results: (weights, means, sigmas) from fitted GMM.
    :param gmm_object: GMM object from either yael or scikit-learn.
    :param ncomponents: Number of components used in Gaussian mixture model.
    :param start_idxs: Start indices of each sliding window.
    :param end_idxs: End indices of each sliding window.
    :return: fv_features: Fisher vector features for mirrored_features.
    """
    print('Generating Fisher vector features...')
    fv_features = np.zeros((len(start_idxs), 2 * ncomponents * np.size(mirrored_features, 1) + ncomponents - 1))
    if not USE_YAEL:
        ws, mus, sigmas = gmm_results
        ncomponents = len(ws)
    for i in range(len(start_idxs)):
        if start_idxs[i] != -1:
            if USE_YAEL:
                X = mirrored_features[int(start_idxs[i]):int(end_idxs[i]) + 1, :].astype('float32')
                fv_features[i, :] = ynumpy.fisher(gmm_results, X, include=['w', 'mu', 'sigma'])
            else:
                X = mirrored_features[int(start_idxs[i]):int(end_idxs[i]) + 1, :]
                num_samples = np.size(X, 0)
                try:
                    gammas = gmm_object.predict_proba(X)
                except:
                    gammas = np.zeros((np.size(X, 0), len(ws)))
                    for obs in range(len(X)):
                        gammas[obs, :] = compute_gmm_probs(X[obs, :], ws, mus, sigmas)
                accus = np.sum(gammas[:, 1:]/ws[1:] - (gammas[:, 0]/ws[0])[:, np.newaxis], axis=0)
                grad_alpha = [accus[idx]/np.sqrt((1/ws[idx+1]+1/ws[0])) for idx in range(0, ncomponents-1)]
                grad_mu = [np.sqrt(sigmas[k, :] / (ws[k])) * np.dot(gammas[:, k], (X - mus[k]) / sigmas[k]) for k in range(ncomponents)]
                grad_sigma = [np.sqrt(1 / (2*ws[k])) * np.dot(gammas[:, k], (X - mus[k])**2 / sigmas[k] - 1) for k in range(ncomponents)]

                fv_features[i, :] = 1/np.sqrt(num_samples)*np.concatenate((grad_alpha, np.array(grad_mu).flatten(), np.array(grad_sigma).flatten()))

    # Normalize
    fv_features = power_l2_normalize(fv_features, power_normalize=False)
    return fv_features


def power_l2_normalize(features, power_normalize=True):
    """
    Sign-square-root and l2 normalize features

    :param features: Features to power and l2-normalize
    :return: features: Power and l2-normalized features
    """
    if power_normalize:
        features = np.sqrt(np.abs(features))*np.sign(features)
    denom = np.linalg.norm(features, axis=-1)
    denom[np.where(denom == 0)] = 1
    features /= denom[:, np.newaxis]
    return features


def compute_gmm_probs(x, ws, mus, sigmas):
    """
    Given a vector x, compute the probability of it belonging to each component of a Gaussian mixture model.

    :param x: Observation to compute probabilities for.
    :param ws: Mixture component probabilities.
    :param mus: Means from mixture model.
    :param sigmas: Diagonals of covariance matrices from mixture model.
    :return: probs: Probability of x belonging to each mixture component.
    """
    ncomponents = len(ws)
    probs = np.zeros(ncomponents)
    denom = np.sum(ws[j] * scipy.stats.multivariate_normal.pdf(x, mus[j], sigmas[j]) for j in range(ncomponents))
    for i in range(ncomponents):
        probs[i] = ws[i] * scipy.stats.multivariate_normal.pdf(x, mus[i], sigmas[i]) / denom
    return probs
