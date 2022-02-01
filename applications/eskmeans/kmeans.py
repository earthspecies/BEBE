"""
A class implementing the K-means algorithm.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016-2017, 2021
"""

import pickle
import numpy as np
import random
import time

DEBUG = 0


#-----------------------------------------------------------------------------#
#                                K-MEANS CLASS                                #
#-----------------------------------------------------------------------------#

class KMeans(object):
    """
    The K-means model class.

    If a component is emptied out, its mean is set to a a random data vector.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D.
    K_max : int
        Maximum number of components.
    assignments : None or str or vector of int
        If vector of int, this gives the initial component assignments. The
        vector should have N entries between 0 and `K_max`. A value of -1
        indicates that a data vector does not belong to any component. If
        `assignments` is None, then all data vectors are left unassigned with
        -1. Alternatively, `assignments` can take one of the following values:
        "rand" assigns data vectors randomly; "each-in-own" assigns each data
        point to a component of its own; and "spread" makes an attempt to
        spread data vectors evenly over the components.

    Attributes
    ----------
    mean_numerators : matrix (K_max, D)
        The sum of all the data vectors assigned to each component, i.e. the
        component means without normalization.
    means : matrix (K_max, D)
        component means, i.e. with normalization.
    counts : vector of int
        Counts for each of the `K_max` components.
    assignments : vector of int
        component assignments for each of the `N` data vectors. 
    """

    def __init__(self, X, K_max, assignments="rand"):

        # Attributes from parameters
        self.X = X
        self.N, self.D = X.shape
        self.K_max = K_max
        self.K = 0

        # Attributes
        self.mean_numerators = np.zeros((self.K_max, self.D), np.float)
        self.counts = np.zeros(self.K_max, np.int)
        self.assignments = -1*np.ones(self.N, dtype=np.int)
        self.setup_random_means()
        self.means = self.random_means.copy()

        # Initial component assignments
        if assignments is not None:
            if isinstance(assignments, str) and assignments == "rand":
                assignments = np.random.randint(0, self.K_max, self.N)
            elif isinstance(assignments, str) and assignments == "each-in-own":
                assignments = np.arange(self.N)
            elif isinstance(assignments, str) and assignments == "spread":
                assignment_list = (
                    list(range(self.K_max))*int(np.ceil(float(self.N)/self.K_max))
                    )[:self.N]
                random.shuffle(assignment_list)
                assignments = np.array(assignment_list)
            else:
                # `assignments` is a vector
                assignments = np.asarray(assignments, np.int)

            # Make sure we have consequetive values
            for k in range(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break

            # Add the data vectors
            for k in range(assignments.max() + 1):
                for i in np.where(assignments == k)[0]:
                    self.add_item(i, k)

    def save(self, f):
        """Pickle necessary attributes to opened file."""
        pickle.dump(self.assignments, f, -1)

    def load(self, f):
        """Load attributes from the opened pickle file and re-initialize."""
        assignments = pickle.load(f)
        self.__init__(self.X, self.K_max, assignments)

    def setup_random_means(self):
        self.random_means = self.X[np.random.choice(range(self.N), self.K_max, replace=True), :]

    def add_item(self, i, k):
        """
        Add data vector `X[i]` to component `k`.

        If `k` is `K`, then a new component is added. No checks are performed
        to make sure that `X[i]` is not already assigned to another component.
        """
        assert not i == -1
        assert self.assignments[i] == -1

        if k > self.K:
            k = self.K
        if k == self.K:
            self.K += 1

        self.mean_numerators[k, :] += self.X[i]
        self.counts[k] += 1
        self.means[k, :] = self.mean_numerators[k, :] / self.counts[k]
        self.assignments[i] = k

    def del_item(self, i):
        """Remove data vector `X[i]` from its component."""
        
        assert not i == -1
        k = self.assignments[i]

        # Only do something if the data vector has been assigned
        if k != -1:
            self.counts[k] -= 1
            self.assignments[i] = -1
            self.mean_numerators[k, :] -= self.X[i]
            if self.counts[k] != 0:
                self.means[k, :] = self.mean_numerators[k, :] / self.counts[k]

    def del_component(self, k):
        """Remove component `k`."""

        assert k < self.K

        if DEBUG > 0:
            print("Deleting component: {}".format(k))
        self.K -= 1
        if k != self.K:
            # Put stats from last component into place of the one being removed
            self.mean_numerators[k] = self.mean_numerators[self.K]
            self.counts[k] = self.counts[self.K]
            self.means[k, :] = self.mean_numerators[self.K, :] / self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k

        # Empty out stats for last component
        self.mean_numerators[self.K].fill(0.)
        self.counts[self.K] = 0
        self.means[self.K] = self.random_means[self.K]

    def clean_components(self):
        """Remove all empty components."""
        for k in np.where(self.counts[:self.K] == 0)[0][::-1]:
            self.del_component(k)

    def neg_sqrd_norm(self, i):
        """
        Return the vector of the negative squared distances of `X[i]` to the
        mean of each of the components.
        """
        deltas = self.means - self.X[i]
        return -(deltas*deltas).sum(axis=1)  # equavalent to np.linalg.norm(deltas, axis=1)**2

    def max_neg_sqrd_norm_i(self, i):
        return np.max(self.neg_sqrd_norm(i))

    def argmax_neg_sqrd_norm_i(self, i):
        return np.argmax(self.neg_sqrd_norm(i))

    def sum_neg_sqrd_norm(self):
        """
        Return the k-means maximization objective: the sum of the negative
        squared norms of all the items.
        """
        objective = 0
        for k in range(self.K):
            X = self.X[np.where(self.assignments == k)]
            mean = self.mean_numerators[k, :]/self.counts[k]
            deltas = mean - X
            objective += -np.sum(deltas*deltas)
        return objective

    def get_assignments(self, list_of_i):
        """
        Return a vector of the current assignments for the data vector indices
        in `list_of_i`.
        """
        return self.assignments[np.asarray(list_of_i)]

    def get_max_assignments(self, list_of_i):
        """
        Return a vector of the best assignments for the data vector indices in
        `list_of_i`.
        """
        return [self.argmax_neg_sqrd_norm_i(i) for i in list_of_i]

    def get_n_assigned(self):
        """Return the number of assigned data vectors."""
        return len(np.where(self.assignments != -1)[0])

    def fit(self, n_iter, consider_unassigned=True):
        """
        Perform `n_iter` iterations of k-means optimization.

        Parameters
        ----------
        consider_unassigned : bool
            Whether unassigned vectors (-1 in `assignments`) should be
            considered during optimization.

        Return
        ------
        record_dict : dict
            Contains several fields describing the optimization iterations.
            Each field is described by its key and statistics are given in a
            list covering the iterations.
        """

        # Setup record dictionary
        record_dict = {}
        record_dict["sum_neg_sqrd_norm"] = []
        record_dict["K"] = []
        record_dict["n_mean_updates"] = []
        record_dict["sample_time"] = []
        
        # Loop over iterations
        start_time = time.time()
        for i_iter in range(n_iter):

            # List of tuples (i, k) where i is the data item and k is the new
            # component to which it should be assigned
            mean_numerator_updates = []

            # Assign data items
            for i in range(self.N):
                
                # Keep track of old value in case we do not have to update
                k_old = self.assignments[i]
                if not consider_unassigned and k_old == -1:
                    continue

                # Pick the new component
                scores = self.neg_sqrd_norm(i)
                k = np.argmax(scores)
                if k != k_old:
                    mean_numerator_updates.append((i, k))

            # Update means
            for i, k in mean_numerator_updates:
                self.del_item(i)
                self.add_item(i, k)

            # Remove empty components
            self.clean_components()

            # Update record
            record_dict["sum_neg_sqrd_norm"].append(self.sum_neg_sqrd_norm())
            record_dict["K"].append(self.K)
            record_dict["n_mean_updates"].append(len(mean_numerator_updates))
            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()

            # Log info
            info = "Iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            print(info)

            if len(mean_numerator_updates) == 0:
                break

        return record_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#


def main():

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    K = 6           # number of components
    n_iter = 10

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Setup K-means model
    model = KMeans(X, K, "rand")


if __name__ == "__main__":
    main()
