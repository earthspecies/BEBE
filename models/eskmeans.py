import yaml
import numpy as np
import pickle
import os
import scipy.signal as signal
import tqdm
from behavior_benchmarks.applications.eskmeans import eskmeans_wordseg
from sklearn.decomposition import PCA

# Utility functions

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
    self.predictions_dict = None
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    self.landmark_hop_size = self.model_config['landmark_hop_size']
    self.n_epochs = self.model_config['n_epochs']
    self.n_landmarks_max = self.model_config['n_landmarks_max'] 
    self.embed_length = self.model_config['embed_length']  
    self.n_clusters = self.model_config['n_clusters']
    self.boundary_init_lambda = self.model_config['boundary_init_lambda']
    
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
    return pca
    
  def fit(self):
    # initialize raw data -> latent variables encoder
    encoder = self.build_encoder()
    
    # Set up data batches
    
    # chop up each track into something computationally tractable
    batch_len = 10000

    all_data_keys = []
    for fp in self.config['train_data_fp']:
        input_data = self.load_model_inputs(fp)
        len_track = np.shape(input_data)[0]
        start_points = list(np.arange(0, len_track, batch_len))
        for start_point in start_points:
            key = '---'.join([fp, str(start_point)])
            all_data_keys.append(key)
            
    current_epoch = 0
    first_batch = True
    
    while current_epoch < self.n_epochs:
      print("Beginning on epoch %d" % current_epoch)
      current_epoch_keys = all_data_keys.copy()
      current_epoch_keys = np.random.permutation(current_epoch_keys)
      current_epoch_keys = list(current_epoch_keys)
      
      # keep track of current epoch progress
      record_dict = {}
      record_dict["sum_neg_sqrd_norm"] = []
      record_dict["sum_neg_len_sqrd_norm"] = []
      record_dict["components"] = []
      record_dict["sample_time"] = []
      record_dict["n_tokens"] = []
      
      for input_data_key in tqdm.tqdm(current_epoch_keys):  
        train_data_dict = {}
        #print("considering %s" % input_data_key)
        input_data = self.load_model_inputs(input_data_key.split('---')[0])

        chunked_input_data = input_data[start_point: start_point + batch_len,:]
        encoded_input_data = encoder.transform(chunked_input_data)
        train_data_dict[key] = encoded_input_data

        #initialize landmarks in short chunks
        landmarks_dict = {}

        for key in train_data_dict:
            num_samples = np.shape(train_data_dict[key])[0]
            boundaries = np.arange(self.landmark_hop_size, num_samples, self.landmark_hop_size)
            boundaries = list(boundaries)
            boundaries.append(num_samples) # unsure if this is necessary?
            landmarks_dict[key] = boundaries

        # Find all the possible segmentation intervals

        seglist_dict = {}
        for key in landmarks_dict:
            seglist_dict[key] = get_seglist_from_landmarks(
                landmarks_dict[key], self.n_landmarks_max
                )

        # Precompute all the embeddings        
        downsample_dict = {}
        #print("downsampling latents")
        downsample_dict = {}
        for key in train_data_dict.keys(): #tqdm
            ### Potential to use different embedding method
            downsample_dict[key] = downsample_track(train_data_dict[key],
                                                        seglist_dict[key],
                                                        downsample_length = self.embed_length)

        # Intermediate variables
        #print("generating intermediate variables")
        lengths_dict = {}
        for i in landmarks_dict:
            lengths_dict[i] = len(landmarks_dict[i])

        vec_ids_dict = get_vec_ids_dict(lengths_dict, self.n_landmarks_max)
        durations_dict = get_durations_dict(landmarks_dict, self.n_landmarks_max)

        ### Now I get to instantiate the model and train

        # Model
        if first_batch:
          print("Initializing randomly")
          init_assignments = "rand"
          init_means = None
          first_batch = False
        else:
          init_assignments = None
          init_means = cluster_means
        
        
        ksegmenter = eskmeans_wordseg.ESKmeans(
            K_max=self.n_clusters,
            embedding_mats=downsample_dict, vec_ids_dict=vec_ids_dict,
            durations_dict=durations_dict, landmarks_dict=landmarks_dict,
            boundary_init_lambda = self.boundary_init_lambda, 
            n_slices_min=0,
            n_slices_max=self.n_landmarks_max,
            min_duration=0,
            init_means = init_means,
            init_assignments=init_assignments,
            wip=0
            )
        
        # Segment
        segmenter_record = ksegmenter.segment(n_iter=1)
        
        for key in record_dict:
          record_dict[key].append(segmenter_record[key])
        
        cluster_means = ksegmenter.acoustic_model.means
        
      #########
      info = "Finished epoch: " + str(current_epoch)
      info += ", sum_neg_sqrd_norm: " + str(sum(record_dict["sum_neg_sqrd_norm"]))
      info += ", sum_neg_len_sqrd_norm: " + str(sum(record_dict["sum_neg_len_sqrd_norm"]))
      info += ", n_tokens: " + str(sum(record_dict["n_tokens"]))
      print(info)
      
      current_epoch +=1
      #########

    self.model = ksegmenter
    
    
    
    #####The following should go in the predict method
    
    # Obtain clusters and landmarks (frame indices)
    unsup_transcript = {}
    unsup_landmarks = {}
    for i_utt in range(self.model.utterances.D):
        utt = self.model.ids_to_utterance_labels[i_utt]
        unsup_transcript[utt] = self.model.get_unsup_transcript_i(i_utt)
        if -1 in unsup_transcript[utt]:
            logger.warning(
                "Unassigned cuts in: " + utt + " (transcript: " +
                str(unsup_transcript[utt]) + ")"
                )
        unsup_landmarks[utt] = (
            self.model.utterances.get_segmented_landmarks(i_utt)
            )
        
    # Assemble a dictionary of predictions, in a compressed format
    # Need to put back together the tracks which we chopped up earlier
    # We get predictions_dict[filepath] is a list of tuples (cluster, start_frame, end_frame)
    # These are adjusted back to the indexing of the original (long) file
    # When calling the predict method, we expand the compressed version out to per-frame predictions

    predictions_dict = {}
    for utt_key in unsup_transcript:
        fp = utt_key.split('---')[0]
        if fp not in predictions_dict:
            predictions_dict[fp] = []
        start_point = int(utt_key.split('---')[1])
        for i, cluster in enumerate(unsup_transcript[utt_key]):
            start, end = unsup_landmarks[utt_key][i]
            start += start_point
            end += start_point
            predictions_dict[fp].append((cluster, start, end))
            
            
    self.predictions_dict = predictions_dict
    ############
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    raise ValueError('predict not implemented, use method predict_from_file')
  
  def predict_from_file(self, fp):
    if self.predictions_dict is not None:
      inputs = self.load_model_inputs(fp)
      predictions = np.full(np.shape(inputs)[0], -1)
      if fp not in self.predictions_dict:
        raise ValueError('file path not included in training data')
      pred_compressed = self.predictions_dict[fp]
      for chunk in pred_compressed:
          cluster, start, end = chunk
          predictions[start: end] = cluster
      return predictions
    
    else:
      raise ValueError('Model is not trained, cannot make predictions yet')