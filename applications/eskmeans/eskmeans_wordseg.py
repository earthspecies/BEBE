"""
Based on:
Embedded segmental K-means model for unsupervised word segmentation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016-2017

I've edited it substantially, so functions outside the behavior benchmarks loop aren't guaranteed to be compatible
"""

from joblib import Parallel, delayed
import numpy as np
import random
import time
import tqdm

from behavior_benchmarks.applications.eskmeans.kmeans import KMeans
from behavior_benchmarks.applications.eskmeans.utterances import Utterances

DEBUG = 0
SEGMENT_DEBUG_ONLY = False
I_DEBUG_MONITOR = 0


#-----------------------------------------------------------------------------#
#                       EMBEDDED SEGMENTAL K-MEANS CLASS                      #
#-----------------------------------------------------------------------------#

class ESKmeans(object):
    """
    Embedded segmental K-means.

    Segmentation and clustering are carried out using this class. Variables
    related to the segmentation are stored in the `utterances` attribute, which
    deals with all utterance-level information but knows nothing about the
    acoustics. The `kmeans` attribute deals with all the acoustic embedding
    operations. In member functions, index `i` generally refers to the index of
    an utterance.

    Parameters
    ----------
    K_max : int
        Maximum number of components.
    embedding_mats : dict of matrix
        The matrices of embeddings for every utterance.
    vec_ids_dict : dict of vector of int
        For every utterance, the vector IDs (see `Utterances`).
    landmarks_dict : dict of list of int
        For every utterance, the landmark points at which word boundaries are
        considered, given in the number of frames (10 ms units) from the start
        of each utterance. There is an implicit landmark at the start of every
        utterance.
    durations_dict : dict of vector of int
        The shape of this dict is the same as that of `vec_ids_dict`, but here
        the duration (in frames) of each of the embeddings are given.
    n_slices_min : int
        The minimum number of landmarks over which an embedding can be
        calculated.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be
        calculated.
    min_duration : int
        Minimum duration of a segment.
    wip : float
        Word insertion penalty.
    p_boundary_init : float
        See `Utterances`.
    boundary_init_lambda: positive float
        See `Utterances`. Added in by BH for dealing with long files
    init_assignments : str
        This setting determines how the initial acoustic model assignments are
        determined: "rand" assigns data vectors randomly; "each-in-own" assigns
        each data point to a component of its own; and "spread" makes an
        attempt to spread data vectors evenly over the components.
    init_means : if init_assignments == None, we can alternatively initialize the model with 
        prototypes before segmenting

    Attributes
    ----------
    utterances : Utterances
        Knows nothing about the acoustics. The indices in the `vec_ids`
        attribute refers to the embedding at the corresponding row in
        `acoustic_model.X`.
    acoustic_model : KMeans
        Knows nothing about utterance-level information. All embeddings are
        stored in this class in its `X` attribute.
    ids_to_utterance_labels : list of str
        Keeps track of utterance labels for a specific utterance ID.
    """

    def __init__(self, K_max, embedding_mats, vec_ids_dict, durations_dict,
            landmarks_dict, processed_embeddings, n_slices_min=0, n_slices_max=20, min_duration=0,
            p_boundary_init=0.5, boundary_init_lambda = None, init_assignments="rand", wip=0, init_means = None):

        # Attributes from parameters
        self.n_slices_min = n_slices_min
        self.n_slices_max = n_slices_max
        self.wip = wip
        
        embeddings, vec_ids, ids_to_utterance_labels = processed_embeddings        
        
        self.ids_to_utterance_labels = ids_to_utterance_labels
        N = embeddings.shape[0]
    
        # Initialize `utterances`
        #print("Initializing utterances")
        lengths = [len(landmarks_dict[i]) for i in ids_to_utterance_labels]
        landmarks = [landmarks_dict[i] for i in ids_to_utterance_labels]
        durations = [durations_dict[i] for i in ids_to_utterance_labels]
        self.utterances = Utterances(
            lengths, vec_ids, durations, landmarks,
            p_boundary_init=p_boundary_init, boundary_init_lambda = boundary_init_lambda, n_slices_min=n_slices_min,
            n_slices_max=n_slices_max, min_duration=min_duration
            )

        # Embeddings in the initial segmentation
        #print("Initializing embeddings")
        init_embeds = []
        for i in range(self.utterances.D):
            init_embeds.extend(self.utterances.get_segmented_embeds_i(i))
        init_embeds = np.array(init_embeds, dtype=int)
        init_embeds = init_embeds[np.where(init_embeds != -1)]
        #print("No. initial embeddings: {}".format(init_embeds.shape[0]))

        # Initialize the K-means components
        #print("Initializing K-means")
        assignments = -1*np.ones(N, dtype=int)
        if init_assignments == "rand":
            assignments[init_embeds] = np.random.randint(0, K_max, len(init_embeds))
        elif init_assignments == "spread":
            n_init_embeds = len(init_embeds)
            assignment_list = (
                list(range(K_max))*int(np.ceil(float(n_init_embeds)/K_max))
                )[:n_init_embeds]
            random.shuffle(assignment_list)
            assignments[init_embeds] = np.array(assignment_list)
        self.acoustic_model = KMeans(embeddings, K_max, assignments, init_means = init_means)

    def save(self, f):
        self.acoustic_model.save(f)
        self.utterances.save(f)

    def load(self, f):
        self.acoustic_model.load(f)
        self.utterances.load(f)

    def segment_i(self, i):
        """
        Segment new boundaries and cluster new segments for utterance `i`.

        Return
        ------
        sum_neg_len_sqrd_norm : float
            The length-weighted K-means objective for this utterance.
        """

        # Debug trace
        if DEBUG > 0:
            print("Segmenting utterance: " + str(i))
            if i == I_DEBUG_MONITOR:
                print("-"*79)
                print("Statistics before sampling")
                print(
                    "sum_neg_sqrd_norm before sampling: " +
                    str(self.acoustic_model.sum_neg_sqrd_norm())
                    )
                print("Unsupervised transcript: " + str(self.get_unsup_transcript_i(i)))
                print("Unsupervised max transcript: " + str(self.get_max_unsup_transcript_i(i)))

        # The embeddings before segmentation
        old_embeds = self.utterances.get_segmented_embeds_i(i)

        # Get the scores of the embeddings
        N = self.utterances.lengths[i]
        vec_embed_neg_len_sqrd_norms = self.get_vec_embed_neg_len_sqrd_norms(
            self.utterances.vec_ids[i, :int((N**2 + N)/2)],
            self.utterances.durations[i, :int((N**2 + N)/2)]
            )

        # Debug trace
        if DEBUG > 0 and i == I_DEBUG_MONITOR:
            print("vec_embed_neg_len_sqrd_norms: " + str(vec_embed_neg_len_sqrd_norms))
            neg_sqrd_norms = [
                self.acoustic_model.max_neg_sqrd_norm_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            print("Embeddings: " + str(embeddings))
            print("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            print("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            print("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            print("neg_sqrd_norms: " + str(neg_sqrd_norms))
            print("neg_len_sqrd_norms: " + str(neg_sqrd_norms*np.array(lengths)))
            print("sum_neg_len_sqrd_norms: " + str(np.sum(neg_sqrd_norms*np.array(lengths))))

        # Draw new boundaries for utterance i
        sum_neg_len_sqrd_norm, self.utterances.boundaries[i, :N] = forward_backward_kmeans_viterbi(
            vec_embed_neg_len_sqrd_norms, N, self.n_slices_min, self.n_slices_max, i
            )
        
        # Debug trace
        if DEBUG > 0 and i == I_DEBUG_MONITOR:
            print("Statistics after sampling, but before adding new embeddings to acoustic model")
            neg_sqrd_norms = [
                self.acoustic_model.max_neg_sqrd_norm_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            print("Embeddings: " + str(embeddings))
            print("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            print("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            print("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            print("neg_sqrd_norms: " + str(neg_sqrd_norms))
            print("neg_len_sqrd_norms: " + str(neg_sqrd_norms*np.array(lengths)))
            print("sum_neg_len_sqrd_norms: " + str(np.sum(neg_sqrd_norms*np.array(lengths))))

        # Remove old embeddings and add new ones; this is equivalent to
        # assigning the new embeddings and updating the means.
        new_embeds = self.utterances.get_segmented_embeds_i(i)
        new_k = self.get_max_unsup_transcript_i(i)

        for i_embed in old_embeds:
            if i_embed == -1:
                continue  # don't remove a non-embedding (would accidently remove the last embedding)
            self.acoustic_model.del_item(i_embed)
        for i_embed, k in zip(new_embeds, new_k):
            self.acoustic_model.add_item(i_embed, k)
        self.acoustic_model.clean_components()

        # Debug trace
        if DEBUG > 0 and i == I_DEBUG_MONITOR:
            print(
                "sum_neg_sqrd_norm after sampling: " +
                str(self.acoustic_model.sum_neg_sqrd_norm())
                )
            print("Unsupervised transcript after sampling: " + str(self.get_unsup_transcript_i(i)))
            print("-"*79)

        return sum_neg_len_sqrd_norm  # technically, this is with the old means (before updating, above)

    def segment(self, n_iter, n_iter_inbetween_kmeans=0):
        """
        Perform segmentation of all utterances and update the K-means model.

        Parameters
        ----------
        n_iter : int
            Number of iterations of segmentation.
        n_iter_inbetween_kmeans : int
            Number of K-means iterations inbetween segmentation iterations.

        Return
        ------
        record_dict : dict
            Contains several fields describing the optimization iterations.
            Each field is described by its key and statistics are given in a
            list covering the iterations.
        """

        # Debug trace
        #print("Segmenting for {} iterations".format(n_iter))
        if DEBUG > 0:
            print(
                "Monitoring utterance {} (index={:d})".format(
                self.ids_to_utterance_labels[I_DEBUG_MONITOR], I_DEBUG_MONITOR)
                )

        # Setup record dictionary
        record_dict = {}
        record_dict["sum_neg_sqrd_norm"] = []
        record_dict["sum_neg_len_sqrd_norm"] = []
        record_dict["components"] = []
        record_dict["sample_time"] = []
        record_dict["n_tokens"] = []

        # Loop over sampling iterations
        for i_iter in range(n_iter):

            start_time = time.time()

            # Loop over utterances
            utt_order = list(range(self.utterances.D))
            random.shuffle(utt_order)
            if SEGMENT_DEBUG_ONLY:
                utt_order = [I_DEBUG_MONITOR]
            sum_neg_len_sqrd_norm = 0
            for i_utt in utt_order:
                sum_neg_len_sqrd_norm += self.segment_i(i_utt)

            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["sum_neg_sqrd_norm"].append(self.acoustic_model.sum_neg_sqrd_norm())
            record_dict["sum_neg_len_sqrd_norm"].append(sum_neg_len_sqrd_norm)
            record_dict["components"].append(self.acoustic_model.K)
            record_dict["n_tokens"].append(self.acoustic_model.get_n_assigned())

            # info = "Iteration: " + str(i_iter)
            # for key in sorted(record_dict):
            #     info += ", " + key + ": " + str(record_dict[key][-1])
            # print(info)

            # Perform intermediate acoustic model re-sampling
            if n_iter_inbetween_kmeans > 0:
                print("WARNING: kmeans fit isn't guaranteed to work in batched implementation")
                self.acoustic_model.fit(
                    n_iter_inbetween_kmeans, consider_unassigned=False
                    )

        return record_dict

    def segment_only_i(self, i):
        """
        Segment new boundaries for utterance `i`, without cluster assignment.

        Although cluster assignments are not updated, the cluster assignments
        are determined and returned (but the `acoustic_model` is not updated).

        Return
        ------
        i, sum_neg_len_sqrd_norm, new_boundaries, old_embeds, new_embeds,
                new_k : (int, vector, float, list, list, list)
            The utterance index; the length-weighted K-means objective for this
            utterance; newly segmented boundaries; embeddings before
            segmentation; new embeddings after segmentation; new embedding
            assignments.
        """

        # Debug trace
        if DEBUG > 0:
            print("Segmenting utterance: " + str(i))
            if i == I_DEBUG_MONITOR:
                print("-"*79)
                print("Statistics before sampling")
                print(
                    "sum_neg_sqrd_norm before sampling: " +
                    str(self.acoustic_model.sum_neg_sqrd_norm())
                    )
                print("Unsupervised transcript: " + str(self.get_unsup_transcript_i(i)))
                print("Unsupervised max transcript: " + str(self.get_max_unsup_transcript_i(i)))

        # The embeddings before segmentation
        old_embeds = self.utterances.get_segmented_embeds_i(i)

        # Get the scores of the embeddings
        N = self.utterances.lengths[i]
        vec_embed_neg_len_sqrd_norms = self.get_vec_embed_neg_len_sqrd_norms(
            self.utterances.vec_ids[i, :int((N**2 + N)/2)],
            self.utterances.durations[i, :int((N**2 + N)/2)]
            )

        # Debug trace
        if DEBUG > 0 and i == I_DEBUG_MONITOR:
            print("vec_embed_neg_len_sqrd_norms: " + str(vec_embed_neg_len_sqrd_norms))
            neg_sqrd_norms = [
                self.acoustic_model.max_neg_sqrd_norm_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            print("Embeddings: " + str(embeddings))
            print("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            print("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            print("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            print("neg_sqrd_norms: " + str(neg_sqrd_norms))
            print("neg_len_sqrd_norms: " + str(neg_sqrd_norms*np.array(lengths)))
            print("sum_neg_len_sqrd_norms: " + str(np.sum(neg_sqrd_norms*np.array(lengths))))

        # Draw new boundaries for utterance i
        sum_neg_len_sqrd_norm, new_boundaries = forward_backward_kmeans_viterbi(
            vec_embed_neg_len_sqrd_norms, N, self.n_slices_min, self.n_slices_max, i
            )
        # sum_neg_len_sqrd_norm, self.utterances.boundaries[i, :N] = forward_backward_kmeans_viterbi(
        #     vec_embed_neg_len_sqrd_norms, N, self.n_slices_min, self.n_slices_max, i
        #     )
        # new_boundaries = self.utterances.boundaries[i, :N]

        # Debug trace
        if DEBUG > 0 and i == I_DEBUG_MONITOR:
            print("Statistics after sampling, but before adding new embeddings to acoustic model")
            neg_sqrd_norms = [
                self.acoustic_model.max_neg_sqrd_norm_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            print("Embeddings: " + str(embeddings))
            print("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            print("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            print("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            print("neg_sqrd_norms: " + str(neg_sqrd_norms))
            print("neg_len_sqrd_norms: " + str(neg_sqrd_norms*np.array(lengths)))
            print("sum_neg_len_sqrd_norms: " + str(np.sum(neg_sqrd_norms*np.array(lengths))))

        # Remove old embeddings and add new ones; this is equivalent to
        # assigning the new embeddings and updating the means.
        # new_embeds = self.utterances.get_segmented_embeds_i(i)
        # new_k = self.get_max_unsup_transcript_i(i)

        new_embeds = self.utterances.get_segmented_embeds_i_bounds(i, new_boundaries)
        new_k = self.get_max_unsup_transcript_i_embeds(i, new_embeds)

        # for i_embed in old_embeds:
        #     if i_embed == -1:
        #         continue  # don't remove a non-embedding (would accidently remove the last embedding)
        #     self.acoustic_model.del_item(i_embed)
        # for i_embed, k in zip(new_embeds, new_k):
        #     self.acoustic_model.add_item(i_embed, k)
        # self.acoustic_model.clean_components()

        # Debug trace
        if DEBUG > 0 and i == I_DEBUG_MONITOR:
            print(
                "sum_neg_sqrd_norm after sampling: " +
                str(self.acoustic_model.sum_neg_sqrd_norm())
                )
            print("Unsupervised transcript after sampling: " + str(self.get_unsup_transcript_i(i)))
            print("-"*79)

        return i, sum_neg_len_sqrd_norm, new_boundaries, old_embeds, new_embeds, new_k

    def segment_parallel(self, n_iter, n_iter_inbetween_kmeans=0, n_cpus=1,
            n_batches=1):
        """
        Perform segmentation of all utterances and update the K-means model.

        Parameters
        ----------
        n_iter : int
            Number of iterations of segmentation.
        n_iter_inbetween_kmeans : int
            Number of K-means iterations inbetween segmentation iterations.
        n_cpus : int
            Number of parallel processes.
        n_batches : int
            Over each batch, an update is made.

        Return
        ------
        record_dict : dict
            Contains several fields describing the optimization iterations.
            Each field is described by its key and statistics are given in a
            list covering the iterations.
        """

        # Debug trace
        print("Segmenting for {} iterations".format(n_iter))
        if DEBUG > 0:
            print(
                "Monitoring utterance {} (index={:d})".format(
                self.ids_to_utterance_labels[I_DEBUG_MONITOR], I_DEBUG_MONITOR)
                )

        # Setup record dictionary
        record_dict = {}
        record_dict["sum_neg_sqrd_norm"] = []
        record_dict["sum_neg_len_sqrd_norm"] = []
        record_dict["components"] = []
        record_dict["sample_time"] = []
        record_dict["n_tokens"] = []

        # Loop over sampling iterations
        for i_iter in range(n_iter):

            start_time = time.time()

            # Determine utterance order
            utt_global_order = range(self.utterances.D)
            random.shuffle(utt_global_order)
            n_batch_size = int(np.ceil(len(utt_global_order)/float(n_batches)))

            # Perform segmentation over batches
            sum_neg_len_sqrd_norm = 0
            for i_batch in range(n_batches):
                utt_order = utt_global_order[n_batch_size*i_batch:n_batch_size*(i_batch + 1)]

                # Segment in parallel
                utt_batches = [utt_order[i::n_cpus] for i in range(n_cpus)]
                updates = Parallel(n_jobs=n_cpus)(delayed(
                    local_segment_only_utts)(self, utts) for utts in utt_batches
                    )

                # Aggregate updates
                updates = [item for sublist in updates for item in sublist]  # flatten
                old_embeds = []
                new_embeds = []
                new_k = []
                for (i_utt, cur_sum_neg_len_sqrd_norm, cur_new_bounds, cur_old_embeds, cur_new_embeds,
                        cur_new_k) in updates:
                    sum_neg_len_sqrd_norm += cur_sum_neg_len_sqrd_norm
                    old_embeds.extend(cur_old_embeds)
                    new_embeds.extend(cur_new_embeds)
                    new_k.extend(cur_new_k)

                    N = self.utterances.lengths[i_utt]
                    self.utterances.boundaries[i_utt, :N] = cur_new_bounds

                # Remove old embeddings and add new ones; this is equivalent to
                # assigning the new embeddings and updating the means.
                for i_embed in old_embeds:
                    if i_embed == -1:
                        continue  # don't remove a non-embedding (would accidently remove the last embedding)
                    self.acoustic_model.del_item(i_embed)
                for i_embed, k in zip(new_embeds, new_k):
                    self.acoustic_model.add_item(i_embed, k)
                self.acoustic_model.clean_components()

            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["sum_neg_sqrd_norm"].append(self.acoustic_model.sum_neg_sqrd_norm())
            record_dict["sum_neg_len_sqrd_norm"].append(sum_neg_len_sqrd_norm)
            record_dict["components"].append(self.acoustic_model.K)
            record_dict["n_tokens"].append(self.acoustic_model.get_n_assigned())

            info = "Iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            print(info)

            # Perform intermediate acoustic model re-sampling
            if n_iter_inbetween_kmeans > 0:
                print("WARNING: kmeans fit isn't guaranteed to work in batched implementation")
                self.acoustic_model.fit(
                    n_iter_inbetween_kmeans, consider_unassigned=False
                    )

        return record_dict

    def get_vec_embed_neg_len_sqrd_norms(self, vec_ids, durations):

        # Get scores
        vec_embed_neg_len_sqrd_norms = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            vec_embed_neg_len_sqrd_norms[i] = self.acoustic_model.max_neg_sqrd_norm_i(
                embed_id
                )

            # Scale log marginals by number of frames
            # if np.isnan(durations[i]):
            if durations[i] == -1:
                vec_embed_neg_len_sqrd_norms[i] = -np.inf
            else:
                vec_embed_neg_len_sqrd_norms[i] *= durations[i]#**self.time_power_term

        return vec_embed_neg_len_sqrd_norms + self.wip

    def get_unsup_transcript_i(self, i):
        """
        Return a list of the current component assignments for the current
        segmentation of `i`.
        """
        return list(self.acoustic_model.get_assignments(self.utterances.get_segmented_embeds_i(i)))

    def get_max_unsup_transcript_i(self, i):
        """
        Return a list of the best components for current segmentation of `i`.
        """
        return self.acoustic_model.get_max_assignments(self.utterances.get_segmented_embeds_i(i))

    def get_max_unsup_transcript_i_embeds(self, i, embeddings):
        """
        Return a list of the best components for the given embeddings of `i`.
        """
        return self.acoustic_model.get_max_assignments(embeddings)


#-----------------------------------------------------------------------------#
#                     FORWARD-BACKWARD INFERENCE FUNCTIONS                    #
#-----------------------------------------------------------------------------#

def forward_backward_kmeans_viterbi(vec_embed_neg_len_sqrd_norms, N,
        n_slices_min=0, n_slices_max=0, i_utt=None):
    """
    Segmental K-means viterbi segmentation of an utterance of length `N` based
    on its `vec_embed_neg_len_sqrd_norms` vector and return a bool vector of
    boundaries.

    Parameters
    ----------
    vec_embed_neg_len_sqrd_norms : N(N + 1)/2 length vector
        For t = 1, 2, ..., N the entries `vec_embed_neg_len_sqrd_norms[i:i + t]`
        contains the log probabilties of sequence[0:t] up to sequence[t - 1:t],
        with i = t(t - 1)/2. If you have a NxN matrix where the upper
        triangular (i, j)'th entry is the log probability of sequence[i:j + 1],
        then by stacking the upper triangular terms column-wise, you get
        vec_embed_neg_len_sqrd_norms`. Written out:
        `vec_embed_neg_len_sqrd_norms` = [neg_len_sqrd_norm(seq[0:1]),
        neg_len_sqrd_norm(seq[0:2]), neg_len_sqrd_norm(seq[1:2]),
        neg_len_sqrd_norm(seq[0:3]), ..., neg_len_sqrd_norm(seq[N-1:N])].
    n_slices_max : int
        If 0, then the full length are considered. This won't necessarily lead
        to problems, since unassigned embeddings would still be ignored since
        their assignments are -1 and the would therefore have a log probability
        of -inf.
    i_utt : int
        If provided, index of the utterance for which to print a debug trace;
        this happens if it matches the global `i_debug_monitor`.

    Return
    ------
    (sum_neg_len_sqrd_norm, boundaries) : (float, vector of bool)
        The `sum_neg_len_sqrd_norm` is the sum of the scores in
        `vec_embed_neg_len_sqrd_norms` for the embeddings for the final
        segmentation.
    """

    n_slices_min_cut = -(n_slices_min - 1) if n_slices_min > 1 else None

    boundaries = np.zeros(N, dtype=bool)
    boundaries[-1] = True
    gammas = np.ones(N)
    gammas[0] = 0.0

    # Forward filtering
    i = 0
    for t in range(1, N):
        if np.all(vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] +
                gammas[:t][-n_slices_max:] == -np.inf):
            gammas[t] = -np.inf
        else:
            gammas[t] = np.max(
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
                gammas[:t][-n_slices_max:n_slices_min_cut]
                )
        i += t

    if DEBUG > 0 and i_utt == I_DEBUG_MONITOR:
        print("gammas: " + str(gammas))

    # Backward segmentation
    t = N
    sum_neg_len_sqrd_norm = 0.
    while True:
        i = int(0.5*(t - 1)*t)
        q_t = (
            vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
            gammas[:t][-n_slices_max:n_slices_min_cut]
            )
        # assert not np.isnan(np.sum(q_t))
        if np.all(q_t == -np.inf):
            if DEBUG > 0:
                print("Only impossible solutions for initial back-sampling for utterance " + str(i_utt))
            # Look for first point where we can actually sample and insert a boundary at this point
            while np.all(q_t == -np.inf):
                t = t - 1
                if t == 0:
                    break  # this is a very crappy utterance
                i = int(0.5*(t - 1)*t)
                q_t = (vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] + gammas[:t][-n_slices_max:])
            if DEBUG > 0:
                print("Backtracked to cut " + str(t))
            boundaries[t - 1] = True  # insert the boundary

        q_t = q_t[::-1]
        k = np.argmax(q_t) + 1
        if n_slices_min_cut is not None:
            k += n_slices_min - 1
        if DEBUG > 0 and i_utt == I_DEBUG_MONITOR:
            print("q_t: " + str(q_t))
            print("argmax q_t: " + str(k))
            print("Embedding neg_len_sqrd_norms: " + str(vec_embed_neg_len_sqrd_norms[i + t - k]))
        sum_neg_len_sqrd_norm += vec_embed_neg_len_sqrd_norms[i + t - k]
        if t - k - 1 < 0:
            break
        boundaries[t - k - 1] = True
        t = t - k

    return sum_neg_len_sqrd_norm, boundaries


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def local_segment_only_utts(this, utterances):
    updates = []
    for i_utt in utterances:
        updates.append(this.segment_only_i(i_utt))
    return updates




#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    embedding_mat1 = np.array([
        [ 1.55329044,  0.82568932,  0.56011276],
        [ 1.10640768, -0.41715366,  0.30323529],
        [ 1.24183824, -2.39021548,  0.02369367],
        [ 1.26094544, -0.27567053,  1.35731148],
        [ 1.59711416, -0.54917262, -0.56074459],
        [-0.4298405 ,  1.39010761, -1.2608597 ]
        ])
    embedding_mat2 = np.array([
        [ 1.63075195,  0.25297823, -1.75406467],
        [-0.59324473,  0.96613426, -0.20922202],
        [ 0.97066059, -1.22315308, -0.37979187],
        [-0.31613254, -0.07262261, -1.04392799],
        [-1.11535652,  0.33905751,  1.85588856],
        [-1.08211738,  0.88559445,  0.2924617 ]
        ])

    # Vector IDs
    n_slices = 3
    vec_ids = -1*np.ones((n_slices**2 + n_slices)/2, dtype=int)
    i_embed = 0
    n_slices_max = 20
    for cur_start in range(n_slices):
        for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
            cur_end += 1
            t = cur_end
            i = t*(t - 1)/2
            vec_ids[i + cur_start] = i_embed
            i_embed += 1

    embedding_mats = {}
    vec_ids_dict = {}
    durations_dict = {}
    landmarks_dict = {}
    embedding_mats["test1"] = embedding_mat1
    vec_ids_dict["test1"] = vec_ids
    landmarks_dict["test1"] = [1, 2, 3]
    durations_dict["test1"] = [1, 2, 1, 3, 2, 1]
    embedding_mats["test2"] = embedding_mat2
    vec_ids_dict["test2"] = vec_ids
    landmarks_dict["test2"] = [1, 2, 3]
    durations_dict["test2"] = [1, 2, 1, 3, 2, 1]

    random.seed(1)
    np.random.seed(1)

    # Initialize model
    K_max = 2
    segmenter = ESKmeans(
        K_max, embedding_mats, vec_ids_dict, durations_dict, landmarks_dict,
        p_boundary_init=1.0, n_slices_max=2
        )

    # with open("/tmp/tmp.pkl", "rb") as f:
    #     segmenter.load(f)

    # Perform inference
    record = segmenter.segment(n_iter=3)

    print("Writing: " + "/tmp/tmp.pkl")
    with open("/tmp/tmp.pkl", "wb") as f:
        segmenter.save(f)


if __name__ == "__main__":
    main()
