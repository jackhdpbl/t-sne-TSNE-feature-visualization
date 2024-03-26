# t-sne-TSNE-feature-visualization
t-SNE (t-distributed Stochastic Neighbor Embedding) is a method for visualizing high-dimensional data in lower dimensions, preserving local structure. It reveals hidden patterns by mapping similar points closer and dissimilar ones farther apart, aiding in data exploration and understanding.
# environment
pip install numpy
pip install matplotlib
pip install scikit-learn
# how to use
Feature and label should be in the form of np.array and correspond one-to-one. 'titlename' is the name you want to give to the t-SNE plot, and 'picname' is the name used for saving.
situation 1：you have save your feature and label as a file.
feature = np.load('feature.npy')
label = np.load('label.npy')
situation 2：you want visualize your feature paralleling with training procedure.
for i in range(iteration):
  for j in range(batchnum):   #process all labels and features for all samples in any batch of all iterations. 
     feature.append(features[j].cpu().detach().numpy())
     label.append(label[j].cpu().detach().numpy())
     feature = np.array(feature)
     label = np.array(label)
