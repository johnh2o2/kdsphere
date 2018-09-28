import numpy as np
from scipy.spatial import cKDTree, KDTree

from .utils import spherical_to_cartesian


class KDSphere(object):
    """KD Tree for Spherical Data, built on scipy's cKDTree

    Parameters
    ----------
    data : array_like, shape (N, 2)
        (lon, lat) pairs measured in radians
    **kwargs :
        Additional arguments are passed to cKDTree
    """
    def __init__(self, data, **kwargs):
        self.data = np.asarray(data)
        self.data3d = spherical_to_cartesian(self.data)
        self.kdtree_ = cKDTree(self.data3d, **kwargs)

    def query(self, data, k=1, eps=0, **kwargs):
        """Query for k-nearest neighbors

        Parameters
        ----------
        data : array_like, shape (N, 2)
            (lon, lat) pairs measured in radians
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.

        Returns
        -------
        d : array_like, float, shape=(N, k)
            The distances to the nearest neighbors
        i : array_like, int, shape=(N, k)
            The indices of the neighbors
        """
        data_3d, r = spherical_to_cartesian(data, return_radius=True)
        dist_3d, ind = self.kdtree_.query(data_3d, k=k, eps=eps, **kwargs)
        dist_2d = 2 * np.arcsin(dist_3d * 0.5 / r)
        return dist_2d, ind

    def query_ball_tree(self, data, r, **kwargs):
        """ Query for matches within ``r`` radians

        Parameters
        ----------
        data : Either a KDTree instance or array_like, shape (N, 2)
            (lon, lat) pairs measured in radians, or another KDSphere (or KDTree)
            instance
        r : float
            Search radius (radians)
        **kwargs:
            Additional arguments passed to scipy.spatial.cKDTree.query_ball_tree

        Returns
        -------
        matches: list of lists
            For each element ``self.data[i]`` of this tree, ``matches[i]`` is
            a list of the indices of its neighbors in ``data``
        """

        other_kdtree = None
        if isinstance(data, (cKDTree, KDTree)):
            other_kdtree = data
        elif isinstance(data, KDSphere):
            other_kdtree = data.kdtree_
        else:
            other_kdtree = KDSphere(data).kdtree_

        return self.kdtree_.query_ball_tree(other_kdtree, r, **kwargs)

    def query_ball_point(self, data, r, **kwargs):
        """ Query for all points within within ``r`` radians of ``data``.

        Parameters
        ----------
        data : tuple or array_like, shape (N, 2)
            (lon, lat) pair(s) measured in radians
        r : float
            Search radius (radians)
        **kwargs:
            Additional arguments passed to ``scipy.spatial.cKDTree.query_ball_point``

        Returns
        -------
        matches: list or list of lists
            If ``data`` is a single (lat, long) pair, ``matches`` is a list
            of indices to neighbors within ``r`` radians. If ``data`` is a list
            of (lat, long) pairs, ``matches`` is a list of lists, and
            ``matches[i]`` is a list of the indices of neighbors within ``r``
            radians from ``data[i]``.
        """


        data_3d = spherical_to_cartesian(np.atleast_2d(data),
                                         return_radius=False)

        results = self.kdtree_.query_ball_point(data_3d, r, **kwargs)

        if np.atleast_2d(data).shape[0] == 1:
            return results[0]

        return results
