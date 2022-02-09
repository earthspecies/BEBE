from sklearn.metrics import homogeneity_score

## various information theoretic metrics
 
def homogeneity(labels_coarse, labels_fine):
  return homogeneity_score(labels_coarse, labels_fine)
