# Implements batched training for H Kamper's ESKmeans code

import yaml
import numpy as np
import pickle
import os
import scipy.signal as signal
import tqdm
from behavior_benchmarks.applications.eskmeans import eskmeans_wordseg
from sklearn.decomposition import PCA

# Utility functions

def group_list(input_list, group_size):
    for i in range(0, len(input_list), group_size):
        yield input_list[i:i+group_size]

def get_vec_ids_dict(lengths_dict, n_landmarks_max):
    """
    Every N(N + 1)/2 length vector `vec_ids` contains all the indices for a
    particular utterance. For t = 1, 2, ..., N the entries `vec_ids[i:i
    + t]` contains the IDs of embedding[0:t] up to embedding[t - 1:t], with i =
    t(t - 1)/2. Written out: `vec_ids` = [embed[0:1], embed[0:2], embed[1:2],
    embed[0:3], ..., embed[N-1:N]].
    """
    vec_ids_dict = {}
    for utt in sorted(lengths_dict.keys()):
        i_embed = 0
        n_slices = lengths_dict[utt]
        vec_ids = -1*np.ones(int((n_slices**2 + n_slices)/2), dtype=int)
        for cur_start in range(n_slices):
            for cur_end in range(cur_start, min(n_slices, cur_start + n_landmarks_max)):
                cur_end += 1
                t = cur_end
                i = t*(t - 1)/2
                vec_ids[int(i + cur_start)] = i_embed
                i_embed += 1
        vec_ids_dict[utt] = vec_ids
    return vec_ids_dict


def get_durations_dict(landmarks_dict, n_landmarks_max):
    durations_dict = {}
    for utt in sorted(landmarks_dict.keys()):
        landmarks = [0,] + landmarks_dict[utt]
        N = len(landmarks)  # should be n_slices + 1
        durations = -1*np.ones(int(((N - 1)**2 + (N - 1))/2), dtype=int)
        j = 0
        for t in range(1, N):
            for i in range(t):
                if t - i > N - 1:
                    j += 1
                    continue
                durations[j] = landmarks[t] - landmarks[i]
                j += 1
        durations_dict[utt] = durations
    return durations_dict
  

# Construct a list of all possible (start, end) positions, where start and end correspond to potential change points
def get_seglist_from_landmarks(landmarks, n_landmarks_max):
    seglist = []
    prev_landmark = 0
    for i in range(len(landmarks)):
        for j in landmarks[i:i + n_landmarks_max]:
            seglist.append((prev_landmark, j))
        prev_landmark = landmarks[i]
    return seglist

# embedding by downsampling
def downsample_track(features, seglist, downsample_length):
    """
    Return the downsampled matrix with each row an embedding for a segment in
    the seglist.
    """
    embeddings = []
    for i, j in seglist:
        y = features[i:j+1, :].T
        y_new = signal.resample(y, downsample_length, axis=1).flatten("C")
        embeddings.append(y_new)
    return np.asarray(embeddings)

class eskmeans():
  def __init__(self, config):
    self.config = config
    self.model_config = config['eskmeans_config']
    self.model = None
    self.metadata = config['metadata']
    self.encoder = None
    # chop up each track into something computationally tractable
    self.max_track_len = self.model_config['max_track_len']
    self.time_power_term = self.model_config['time_power_term'] ## Positive float. when 1., we get standard behavior. when <1, we penalize making short segments
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    self.landmark_hop_size = self.model_config['landmark_hop_size']
    self.n_epochs = self.model_config['n_epochs']
    self.n_landmarks_max = self.model_config['n_landmarks_max'] 
    self.embed_length = self.model_config['embed_length']  
    self.n_clusters = self.model_config['n_clusters']
    self.boundary_init_lambda = self.model_config['boundary_init_lambda']
    self.batch_size = self.model_config['batch_size']
    
  def load_model_inputs(self, filepath):
    return np.load(filepath)[:, self.cols_included]
  
  def build_encoder(self):
    ## get data. assume stored in memory for now
    print("Computing whitening transform")
    train_fps = self.config['train_data_fp']
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    
    pca = PCA(whiten = True)
    pca.fit(train_data)
    self.encoder = pca
    
  def process_embeddings(self, embedding_mats, vec_ids_dict, use_temp = True):
      """
      Process the embeddings and vector IDs into single data structures.
      use_temp: whether to use temporary files to speed up computation (for use in main training loop)

      Return
      ------
      (embeddings, vec_ids, utterance_labels_to_ids) : 
              (matrix of float, list of vector of int, list of str)
          All the embeddings are returned in a single matrix, with a `vec_id`
          vector for every utterance and a list of str indicating which `vec_id`
          goes with which original utterance label.
      """

      embeddings = []
      vec_ids = []
      ids_to_utterance_labels = []
      i_embed = 0
      n_disregard = 0

      # Loop over utterances
      for utt in sorted(embedding_mats): #tqdm
          ids_to_utterance_labels.append(utt)
          cur_vec_ids = vec_ids_dict[utt].copy()
          
          # Add to the array of embeddings
          embeddings.append(np.asarray(embedding_mats[utt]))
          
          # Save off the results of this double for loop. Speeds up subsequent epochs by >2x
          utt_temp_filename = '-'.join([str(utt).split('/')[-1].split('.')[0], str(utt).split('---')[-1], 'temp.npy'])
          utt_temp_fp = os.path.join(self.config['temp_dir'], utt_temp_filename)
          
          if use_temp:
            if not os.path.exists(utt_temp_fp):
              # if it's the first time looking at this utterance, we compute the associated cur_vec_ids (bottlenecK)
              i_embed_single = 0
              for i_row, row in enumerate(embedding_mats[utt]):
                # Update vec_ids_dict so that the index points to i_embed
                cur_vec_ids[np.where(vec_ids_dict[utt] == i_row)[0]] = i_embed_single          
                i_embed_single += 1
              np.save(utt_temp_fp, cur_vec_ids)

            cur_vec_ids = np.load(utt_temp_fp)

            #correct to account for batch indexing
            if i_embed != 0:
              cur_vec_ids[cur_vec_ids != -1] = cur_vec_ids[cur_vec_ids != -1] + i_embed
            i_embed += len(embedding_mats[utt])
          
          else:
            for i_row, row in enumerate(embedding_mats[utt]):
                # Update vec_ids_dict so that the index points to i_embed
                cur_vec_ids[np.where(vec_ids_dict[utt] == i_row)[0]] = i_embed           
                i_embed += 1
          
          # Add the updated entry in vec_ids_dict to the overall vec_ids list
          vec_ids.append(cur_vec_ids)
      
      embeddings = np.concatenate(embeddings, axis = 0)
      return embeddings, vec_ids, ids_to_utterance_labels
  
  def prepare_intermediate_variables(self, input_data_keys, input_data_single_file = None, use_temp = True):
    ### input_data_keys is a list of str of the form "fp---start_point"
    ### if input_data_single_file is specified (as a single numpy array), we use that array as input_data instead of loading from file
    train_data_dict = {}
    landmarks_dict = {}
    seglist_dict = {}
    downsample_dict = {}
    lengths_dict = {}
    
    for input_data_key in input_data_keys:
    #print("considering %s" % input_data_key)
      if input_data_single_file is None:
        input_data = self.load_model_inputs(input_data_key.split('---')[0])
      else:
        input_data = input_data_single_file
      start_point = int(input_data_key.split('---')[1])
      chunked_input_data = input_data[start_point: start_point + self.max_track_len,:]
      encoded_input_data = self.encoder.transform(chunked_input_data)
      train_data_dict[input_data_key] = encoded_input_data

    #initialize landmarks in short chunks

    for key in train_data_dict:
        num_samples = np.shape(train_data_dict[key])[0]
        boundaries = np.arange(self.landmark_hop_size, num_samples, self.landmark_hop_size)
        boundaries = list(boundaries)
        boundaries.append(num_samples) # unsure if this is necessary?
        landmarks_dict[key] = boundaries

    # Find all the possible segmentation intervals

    
    for key in landmarks_dict:
        seglist_dict[key] = get_seglist_from_landmarks(
            landmarks_dict[key], self.n_landmarks_max
            )

    #print("downsampling latents")
    
    for key in train_data_dict.keys(): #tqdm
        ### Potential to use different embedding method
        downsample_dict[key] = downsample_track(train_data_dict[key],
                                                    seglist_dict[key],
                                                    downsample_length = self.embed_length)
    #intermediate variables
    for i in landmarks_dict:
        lengths_dict[i] = len(landmarks_dict[i])

    vec_ids_dict = get_vec_ids_dict(lengths_dict, self.n_landmarks_max)
    durations_dict = get_durations_dict(landmarks_dict, self.n_landmarks_max)
    
    # Process embeddings into a single matrix, and vec_ids into a list (entry for each utterance)
    # print("Processing embeddings")
    processed_embeddings = self.process_embeddings(
        downsample_dict, vec_ids_dict, use_temp = use_temp
        )
    
    return downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings
    
  def fit(self):
    # initialize raw data -> latent variables encoder
    self.build_encoder()
    
    # Set up data batches

    all_data_keys = []
    total_samples = 0
    for fp in self.config['train_data_fp']:
        input_data = self.load_model_inputs(fp)
        len_track = np.shape(input_data)[0]
        total_samples += len_track
        start_points = list(np.arange(0, len_track, self.max_track_len))
        for start_point in start_points:
            key = '---'.join([fp, str(start_point)])
            all_data_keys.append(key)
            
    current_epoch = 0
    previous_means = None
    
    while current_epoch < self.n_epochs:
      print("Beginning on epoch %d" % current_epoch)
      current_epoch_keys = all_data_keys.copy()
      current_epoch_keys = np.random.permutation(current_epoch_keys)
      current_epoch_keys = list(current_epoch_keys)
      
      epoch_mean_numerators = []
      epoch_counts = []
      
      # keep track of current epoch progress
      record_dict = {}
      record_dict["sum_neg_sqrd_norm"] = []
      record_dict["sum_neg_len_sqrd_norm"] = []
      record_dict["components"] = []
      record_dict["sample_time"] = []
      record_dict["n_tokens"] = []
      
      for input_data_keys in tqdm.tqdm(list(group_list(current_epoch_keys, self.batch_size))): #tqdm.tqdm(current_epoch_keys):  
        
        downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings = self.prepare_intermediate_variables(input_data_keys)
        
        ### Since this is batched implementation, we repeatedly have to initialize the ESKmeans class
        ### written by H Kamper and pass in the currently discovered cluster means
        if previous_means is None:
          print("Initializing randomly")
          first_batch = False
          ksegmenter = eskmeans_wordseg.ESKmeans(
              K_max=self.n_clusters,
              embedding_mats=downsample_dict, vec_ids_dict=vec_ids_dict,
              durations_dict=durations_dict, landmarks_dict=landmarks_dict, processed_embeddings = processed_embeddings,
              boundary_init_lambda = self.boundary_init_lambda, 
              n_slices_min=0,
              n_slices_max=self.n_landmarks_max,
              min_duration=0,
              init_means = previous_means,
              init_assignments="rand",
              time_power_term = self.time_power_term,
              wip=0
              )

          # use these means for subsequenct batches
          previous_means = ksegmenter.acoustic_model.means.copy()
        
        else:
          ksegmenter = eskmeans_wordseg.ESKmeans(
              K_max=self.n_clusters,
              embedding_mats=downsample_dict, vec_ids_dict=vec_ids_dict,
              durations_dict=durations_dict, landmarks_dict=landmarks_dict, processed_embeddings = processed_embeddings,
              boundary_init_lambda = self.boundary_init_lambda, 
              n_slices_min=0,
              n_slices_max=self.n_landmarks_max,
              min_duration=0,
              init_means = previous_means.copy(),
              init_assignments=None,
              time_power_term = self.time_power_term,
              wip=0
              )
        
        # Segment & update means
        segmenter_record = ksegmenter.segment(n_iter=1)
        for k in record_dict:
          record_dict[k].append(segmenter_record[k][0])
        
        new_mean_numerators = ksegmenter.acoustic_model.mean_numerators.copy()
        new_counts = ksegmenter.acoustic_model.counts.copy()
        
        epoch_mean_numerators.append(new_mean_numerators)
        epoch_counts.append(new_counts)
      
      #do a weighted average to find epoch means
      epoch_counts = sum(epoch_counts)
      epoch_counts = np.expand_dims(epoch_counts, -1)
      epoch_mean_numerators = sum(epoch_mean_numerators)
      epoch_new_means = np.divide(epoch_mean_numerators, epoch_counts, out=ksegmenter.acoustic_model.random_means.copy(), where = epoch_counts != 0)
           
      ksegmenter.acoustic_model.means = epoch_new_means.copy()
      previous_means = epoch_new_means.copy()
      
      
      #########
      info = "Finished epoch: " + str(current_epoch)
      info += ", sum_neg_sqrd_norm: " + str(sum(record_dict["sum_neg_sqrd_norm"]))
      info += ", sum_neg_len_sqrd_norm: " + str(sum(record_dict["sum_neg_len_sqrd_norm"]))
      info += ", n_tokens: " + str(sum(record_dict["n_tokens"]))
      print(info)
      
      current_epoch +=1
      #########

    self.model = ksegmenter
    
    ############
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
      
  def predict_from_intermediates(self, input_data, downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings, start_points):
    ### Since this is batched implementation, we repeatedly have to initialize the ESKmeans class,
    ### that was written by H Kamper, and pass in the currently discovered cluster means

    init_assignments = None
    init_means = self.model.acoustic_model.means.copy() ### lower step size as epoch increases

    ksegmenter = eskmeans_wordseg.ESKmeans(
        K_max=self.n_clusters,
        embedding_mats=downsample_dict, vec_ids_dict=vec_ids_dict,
        durations_dict=durations_dict, landmarks_dict=landmarks_dict, processed_embeddings = processed_embeddings,
        boundary_init_lambda = self.boundary_init_lambda, 
        n_slices_min=0,
        n_slices_max=self.n_landmarks_max,
        min_duration=0,
        init_means = init_means,
        init_assignments=init_assignments,
        time_power_term = self.time_power_term,
        wip=0
        )
    
    
    # Obtain clusters and landmarks (frame indices)
    unsup_transcript = {}
    unsup_landmarks = {}
    for i_utt in range(ksegmenter.utterances.D):
      i, sum_neg_len_sqrd_norm, new_boundaries, old_embeds, new_embeds, new_k = ksegmenter.segment_only_i(i_utt)

      
      ksegmenter.segment_i(i_utt) # this re-segments the track and updates the model, including the means
      ksegmenter.acoustic_model.means = init_means.copy() # re-initialize with our trained means to undo part of the segment_i call
      utt = ksegmenter.ids_to_utterance_labels[i_utt]
      unsup_transcript[utt] = ksegmenter.get_max_unsup_transcript_i(i_utt) # map segments to their appropriate clusters
      if -1 in unsup_transcript[utt]:
        logger.warning(
            "Unassigned cuts in: " + utt + " (transcript: " +
            str(unsup_transcript[utt]) + ")"
            )
      unsup_landmarks[utt] = (
            ksegmenter.utterances.get_segmented_landmarks(i_utt)
            )

    # Assemble a list of predictions, in a compressed format
    # Need to put back together the tracks which we chopped up earlier
    # We get predictions_compressed is a list of tuples (cluster, start_frame, end_frame)
    # These are adjusted back to the indexing of the original (long) file
    # When calling the predict method, we expand the compressed version out to per-frame predictions

    predictions_compressed = []
    for utt_key in unsup_transcript:
        start_point = int(utt_key.split('---')[1])
        for i, cluster in enumerate(unsup_transcript[utt_key]):
            start, end = unsup_landmarks[utt_key][i]
            start += start_point
            end += start_point
            predictions_compressed.append((cluster, start, end))
            
    # Finally put it all together into per-frame predictions

    predictions = np.full(np.shape(input_data)[0], -1)
    for chunk in predictions_compressed:
        cluster, start, end = chunk
        predictions[start: end] = cluster
    return predictions
  
  def predict(self, input_data):

    all_data_keys = []
    len_track = np.shape(input_data)[0]
    start_points = list(np.arange(0, len_track, self.max_track_len))
    fp = 'input' # placeholder for consistency
    for start_point in start_points:
        key = '---'.join([fp, str(start_point)])
        all_data_keys.append(key)

    downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings = self.prepare_intermediate_variables(all_data_keys, input_data_single_file = input_data, use_temp = False)
    
    return self.predict_from_intermediates(input_data, downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings, start_points)
  
  def predict_from_file(self, fp):
    #inputs = self.load_model_inputs(fp)
    #predictions = self.predict(inputs)
    #return predictions
    
    # faster to do this, in case temp files are already generated:
    all_data_keys = []
    input_data = self.load_model_inputs(fp)
    len_track = np.shape(input_data)[0]
    start_points = list(np.arange(0, len_track, self.max_track_len))
    for start_point in start_points:
        key = '---'.join([fp, str(start_point)])
        all_data_keys.append(key)

    downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings = self.prepare_intermediate_variables(all_data_keys)

    return self.predict_from_intermediates(input_data, downsample_dict, vec_ids_dict, durations_dict, landmarks_dict, processed_embeddings, start_points)
  
  