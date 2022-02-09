"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014-2017, 2021
"""

import pickle
import numpy as np
import tqdm


class Utterances(object):
    """
    A corpus of utterances on which acoustic word segmentation is performed.

    This class deals with all utterance segmentation variables and statistics,
    but none of the acoustics (there are dealt with in `KMeans`).

    Parameters
    ----------
    lengths : vector of int
        The lengths for each utterance, given as the number of embedding
        slices.
    vec_ids : list of vector of int
        The `vec_ids` attribute below is initialized from this.
    durations : list of vector of int
        See `ESKmeans`; this is the list equivalent with an entry for every
        utterance.
    landmarks : list of vector of int
        See `ESKmeans`; this is the list equivalent with an entry for every
        utterance.
    init_boundaries : list of list of int or None
        If provided, the initial boundaries are set accordingly.
    p_boundary_init : float
        The initial segmentation boundary probability. Setting this to 0 means
        no initial boundaries, 0.5 means 50% chance of a boundary at each
        landmark location and 1 means boundaries at all locations.
    n_slices_min : int
        See `ESKmeans`. Here it is just used for boundary initialization to
        make sure that initial boundaries are not shorter than this.
    n_slices_max : int
        See `ESKmeans`. Here it is just used for boundary initialization to
        make sure that initial boundaries are not longer than this.
    min_duration : int
        Minimum duration of a segment.

    Attributes
    ----------
    D : int
        The number of utterances.
    N_max : int 
        The maximum length (in landmarks) of an utterance. This is used to
        pre-allocate space.
    vec_ids : matrix (D, N_max(N_max + 1)/2) of int
        The indices of the embeddings between different landmarks. Every N(N +
        1)/2 length row vector contains all the indices for a particular
        utterance. For t = 1, 2, ..., N the entries `vec_ids[i:i + t]` contains
        the IDs of embedding[0:t] up to embedding[t - 1:t], with i = t(t -
        1)/2. Written out: `vec_ids` = [embed[0:1], embed[0:2], embed[1:2],
        embed[0:3], ..., embed[N-1:N]].
    boundaries : matrix (D, N_max) of bool
        Represents the locations of hypothesized word boundaries, with value
        True in the i'th position indicating a boundary after the i'th landmark
        in the sequence. There is a True at the final position, but possibly
        not at the first. E.g. a True at positions 5 and 10 indicates that
        embed[6:11] is a word under the current hypothesis.
    """

    def __init__(self, lengths, vec_ids, durations, landmarks,
            init_boundaries=None, p_boundary_init=0.5, boundary_init_lambda = None, n_slices_min=0,
            n_slices_max=6, min_duration=0):

        # Check parameters
        assert lengths == [len(i) for i in landmarks]

        # Attributes from parameters
        self.lengths = lengths
        self.D = len(lengths)
        assert self.D == len(vec_ids)
        self.N_max = max(lengths)
        self.landmarks = landmarks

        # Initialize `vec_ids` and `durations`
        self.vec_ids = -1*np.ones(
            (self.D, int(self.N_max*(self.N_max + 1)/2)), dtype=np.int32
            )
        for i_vec_id, vec_id in enumerate(vec_ids):
            self.vec_ids[i_vec_id, :len(vec_id)] = vec_id
        # self.durations = -np.nan*np.ones((self.D, self.N_max*(self.N_max + 1)/2), dtype=np.int32)
        # self.durations = np.zeros((self.D, self.N_max*(self.N_max + 1)/2), dtype=np.int32)
        self.durations = -1*np.ones(
            (self.D, int(self.N_max*(self.N_max + 1)/2)), dtype=np.int32
            )
        for i_duration_vec, duration_vec in enumerate(durations):
            if not (min_duration == 0 or len(duration_vec) == 1):
                cur_duration_vec = np.array(duration_vec, dtype=np.float)
                # cur_duration_vec[cur_duration_vec < min_duration] = -np.nan
                cur_duration_vec[cur_duration_vec < min_duration] = -1
                # if np.all(np.isnan(cur_duration_vec)):
                if np.all(cur_duration_vec == -1):
                    cur_duration_vec[np.argmax(duration_vec)] = np.max(duration_vec)
                duration_vec = cur_duration_vec
            self.durations[i_duration_vec, :len(duration_vec)] = duration_vec

        # Initialize `boundaries`
        self.boundaries = np.zeros((self.D, self.N_max), dtype=bool)
        if init_boundaries is not None:
            print("Initializing boundaries from given values")
            for i_utt, bounds in enumerate(init_boundaries):
                landmark = self.landmarks[i_utt]
                closest_landmarks = []
                for bound in bounds:
                    delta = [abs(bound - lm) for lm in landmark]
                    ind = np.argmin(delta)
                    closest_landmarks.append(ind)
                self.boundaries[i_utt, closest_landmarks] = True
                
        elif boundary_init_lambda is not None:
            #print("Initializing boundaries using a random walk")
            # The default random initialization is not practical for long files
            # Alternatively we initialize using a random walk
            for i in range(self.D): #iterate through utterances tqdm
                N = self.lengths[i]
                current_index = -1
                while current_index < N-1:
                  hop_size_to_new_index = np.minimum(n_slices_max, n_slices_min + np.random.poisson(lam = boundary_init_lambda) + 1)
                  new_index = np.minimum(current_index + hop_size_to_new_index, N-1)
                  self.boundaries[i, new_index] = True
                  current_index = new_index
                self.boundaries[i, N-1] = True
            
        elif p_boundary_init == 0:
            print("Initializing boundaries at start and end of utterance")
            # Some constraints are placed below to get valid embeddings from
            # the random initial segmentation, but these are relaxed when
            # p_boundary_init is 0
            for i in range(self.D):
                N = self.lengths[i]
                self.boundaries[i, N - 1] = True
        else:
            print("Initializing boundaries randomly with probability {}".format(p_boundary_init))
            for i in range(self.D):
                N = self.lengths[i]
                while True:

                    self.boundaries[i, 0:N] = (np.random.rand(N) < p_boundary_init)
                    self.boundaries[i, N - 1] = True

                    # Don't allow all disregarded embeddings for initialization
                    if np.all(np.asarray(self.get_segmented_embeds_i(i)) == -1):
                        continue

                    # Test that `n_slices_max` is not exceeded
                    indices = self.get_segmented_landmark_indices(i)
                    if ((np.max([j[1] - j[0] for j in indices]) <= n_slices_max and
                            np.min([j[1] - j[0] for j in indices]) >= n_slices_min) or
                            (N <= n_slices_min)):
                        break

    def save(self, f):
        """Pickle necessary attributes to opened file."""
        pickle.dump(self.boundaries, f, -1)

    def load(self, f):
        """Load attributes from the opened pickle file."""
        self.boundaries = pickle.load(f)

    def get_segmented_embeds_i(self, i):
        """
        Return a list of embedding IDs according to the current segmentation
        for utterance `i`.
        """
        embed_ids = []
        j_prev = 0
        for j in range(self.lengths[i]):
            if self.boundaries[i, j]:
                # We aim to extract seq[j_prev:j+1]. Let the location of this
                # ID be `vec_ids[i, k]`, and we need to find k.
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in `vec_ids[i]`
                k += j_prev  # this is the index of the seq[j_prev:j] in `vec_ids[i]`
                embed_ids.append(self.vec_ids[i, k])
                j_prev = j + 1
        return embed_ids

    def get_segmented_embeds_i_bounds(self, i, boundaries):
        """
        Return a list of embedding IDs according to the given segmentation
        for utterance `i`.
        """
        embed_ids = []
        j_prev = 0
        for j in range(self.lengths[i]):
            if boundaries[j]:
                # We aim to extract seq[j_prev:j+1]. Let the location of this
                # ID be `vec_ids[i, k]`, and we need to find k.
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in `vec_ids[i]`
                k += j_prev  # this is the index of the seq[j_prev:j] in `vec_ids[i]`
                embed_ids.append(self.vec_ids[i, k])
                j_prev = j + 1
        return embed_ids

    def get_segmented_durations_i(self, i):
        """
        Return a list of durations for the current segmentation of utterance
        `i`; this matches up with the embeddings from `get_segmented_embeds_i`.
        """
        durations = []
        j_prev = 0
        for j in range(self.lengths[i]):
            if self.boundaries[i, j]:
                # We aim to extract seq[j_prev:j+1]. Let the location of this
                # ID be `vec_ids[i, k]`, and we need to find k.
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in `vec_ids[i]`
                k += j_prev  # this is the index of the seq[j_prev:j] in `vec_ids[i]`
                durations.append(self.durations[i, k])
                j_prev = j + 1
        return durations

    def get_original_segmented_embeds_i(self, i):
        """
        Return the same list as `get_segmented_embeds_i`, but using
        utterance-specific embedding IDs, starting at 0 for this utterance.

        This is useful for listening to particular segmentations since the
        indices should correspond to the line in the segmentation list files
        used to generate the dense embeddings.
        """
        vec_ids = self.vec_ids[i]
        vec_ids_min = np.min(vec_ids[np.where(vec_ids != -1)])
        return list(self.get_segmented_embeds_i(i) - vec_ids_min)

    def get_segmented_landmark_indices(self, i):
        """
        Return a list of tuple, where every tuple is the start (inclusive) and
        end (exclusive) landmark index for the segmented embeddings.
        """
        indices = []
        j_prev = 0
        for j in np.where(self.boundaries[i][:self.lengths[i]])[0]:
            indices.append((j_prev, j + 1))
            j_prev = j + 1
        return indices

    def get_segmented_landmarks(self, i):
        """
        Return a list of tuple, where every tuple is the start (inclusive) and
        end (exclusive) frame index (landmark) for the segmented embeddings.
        """
        assert not self.landmarks is None
        indices = []
        j_prev = 0
        for _, j in self.get_segmented_landmark_indices(i):
            indices.append((j_prev, self.landmarks[i][j - 1]))
            j_prev = self.landmarks[i][j - 1]
        return indices
