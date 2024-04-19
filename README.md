# PointCloudSegmentation

Semantic segmentation of a point cloud dataset using deep learning

## 1. Existing Methods for Point Cloud Segmentation

- Appr 1 - Point Net
  - Limitations: Lack of local context, global feature depends on absolute coordinate (hard to generalize)
  - https://arxiv.org/pdf/1612.00593.pdf
  - [https://github.com/charlesq34/pointnet](https://github.com/charlesq34/pointnet.git)
  - Overview:
    1. **Per-point Feature Extraction:** Extract individual point features using shared MLP networks.
    2. **Global Feature Learning:** Aggregate point features using max pooling to obtain a global shape descriptor.
    3. **Local and Global Feature Concatenation:** Combine global and local features for each point to incorporate both local and global information.
    4. **Segmentation:** Use the combined features to predict per-point semantic labels, achieving point-level segmentation.
- Appr 2 - PointNet++:
  - Point Net on point neighborhoods rather than entire cloud
  - https://github.com/charlesq34/pointnet2
  - https://arxiv.org/pdf/1706.02413.pdf
  - Overview
    1. **Hierarchical Feature Learning:**
       - **Set Abstraction Levels:** PointNet++ employs a series of set abstraction levels to progressively extract features at increasing contextual scales.
       - **Sampling Layer:** Farthest point sampling (FPS) is used to select a subset of points as centroids for local regions, ensuring efficient coverage of the point cloud.
       - **Grouping Layer:** Points within a defined radius around each centroid are grouped to form local regions.
       - **PointNet Layer:** A mini-PointNet network processes each local region to encode local geometric patterns into feature vectors.
    2. **Density Adaptive PointNet Layers:**
       - **Addressing Non-Uniform Densities:** PointNet++ tackles the challenge of variable point densities by incorporating mechanisms to adaptively combine features from multiple scales.
       - **Multi-scale Grouping (MSG):** Local regions are grouped at different scales, and features from each scale are extracted and concatenated. Random input dropout during training encourages the network to learn optimal feature combinations for varying densities.
       - **Multi-resolution Grouping (MRG):** Features from sub-regions at a lower level are combined with features directly extracted from the current region's raw points, allowing for efficient adaptive feature aggregation based on local density.
    3. **Point Feature Propagation:**
       - For tasks like semantic segmentation, PointNet++ propagates features from subsampled points back to the original point set using distance-based interpolation and skip connections, enabling per-point predictions.
- Appr 3 - Point Convolution as Graph Convolution
  - **Continuous Convolution Approximation:** PointConv treats convolutional filters as continuous functions of local 3D point coordinates relative to a reference point. These functions are then approximated using multi-layer perceptrons (MLPs).
  - https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf
  - Overview:
    1. **PointConv Feature Learning:** Extract point features using PointConv layers, which learn continuous weight and density functions to approximate convolutions on point clouds.
    2. **Hierarchical Feature Encoding:** Similar to PointNet++, use a hierarchical structure with PointConv layers to capture features at different scales.
    3. **PointDeconv Feature Propagation:** Apply PointDeconv layers to propagate features back to the original resolution, combining interpolation and PointConv operations.
    4. **Segmentation:** Predict per-point semantic labels using the propagated features, achieving high-quality segmentation results.
- Appr 4 - VoxelNet:
  - Space partitions into voxels, points within each voxel are transformed to a vector. Space is represented as a sparse 4D tensor, which is processed by convolutional layers.
  - https://arxiv.org/pdf/1711.06396.pdf
  - https://github.com/ModelBunker/VoxelNet-PyTorch
  - Overview:
    1. **Voxel Representation:**
       - **Voxelization:** The point cloud is divided into equally spaced 3D voxels, providing a structured representation while preserving spatial information.
       - **Point-wise Feature Learning:** Points within each voxel are processed through a series of Voxel Feature Encoding (VFE) layers.
       - **VFE Layer:** This novel layer combines point-wise features with a locally aggregated feature (obtained through element-wise max pooling) to capture local 3D shape information within each voxel.
       - **Sparse Tensor Representation:** Only non-empty voxels are considered, resulting in a sparse 4D tensor that reduces memory usage and computational cost.
    2. **Convolutional Middle Layers:**
       - 3D convolutions are applied to the sparse tensor to aggregate features from neighboring voxels, incorporating contextual information into the representation.
    3. **Region Proposal Network (RPN):**
       - A modified RPN architecture takes the output of the convolutional middle layers as input and predicts 3D bounding boxes for objects.
       - The network utilizes a series of convolutional blocks followed by upsampling and concatenation to generate a high-resolution feature map.
       - This feature map is then used to predict both the probability score and the regression targets for object bounding boxes.
- Appr 5 - Vector Neurons:
  - Tackles the problem of rotational invariance. Neurons are not scalars but 3D vectors.1
  - https://arxiv.org/pdf/2104.12229.pdf
  - https://github.com/FlyingGiraffe/vnn
- Appr 6 - Point Cloud Transfomers:
  - https://arxiv.org/pdf/2111.14819.pdf
  - https://github.com/lulutang0608/Point-BERT
  - Overview:
    1. **Point Tokenization:**
       - **Patching:** The point cloud is divided into local patches (sub-clouds) to act as basic units containing meaningful geometric information.
       - **Point Embeddings:** A mini-PointNet network extracts features from each patch, creating point embeddings.
       - **Discrete Variational Autoencoder (dVAE):** A dVAE is trained to learn a vocabulary of discrete point tokens representing various geometric patterns. This allows conversion of point embeddings into a sequence of point tokens.
    2. **Masked Point Modeling (MPM):**
       - **Masking:** Similar to BERT's masked language modeling, Point-BERT masks out portions of the input point cloud (specifically, point embeddings of patches).
       - **Prediction:** The Transformer encoder is tasked with predicting the original point tokens corresponding to the masked regions, learning to infer missing geometric structures.
       - **Supervision:** The dVAE's decoder provides the ground truth point tokens for supervision during pre-training.
    3. **Auxiliary Task - Point Patch Mixing:**
       - To enhance semantic understanding, Point-BERT introduces an auxiliary task inspired by CutMix.
       - Virtual samples are created by mixing sub-clouds from different point clouds, encouraging the model to learn high-level semantics.
       - A contrastive loss is employed to ensure feature similarity between the mixed samples and their original counterparts.
- Appr 7 - NON DL - clustering on patches:
  - Cloud segmented into patches using Dynamic Region Growth, then clustered.
  - Uses normals (which are not available in our dataset)
  - https://cg.cs.tsinghua.edu.cn/papers/TVCG-2019-Semantic.pdf
  - overview:
    1. **Patch Generation:** The point cloud is segmented into patches using a Dynamic Region Growing (DRG) algorithm, which dynamically updates patch shape features during the growing process, making it robust to noise and sampling variations.
    2. **Patch Clustering:** Patches are clustered based on their feature similarity using an unsupervised clustering method (K-means++). This helps to represent diverse patch characteristics within each object category and learn contextual relationships more robustly.
    3. **Patch Context Analysis:** Pairwise relationships between adjacent patches are analyzed, considering factors like distance, orientation, and labels. This information is used to learn contextual rules that guide semantic labeling.
    4. **Semantic Labeling:** In the test stage, the point cloud is segmented into patches and each patch is assigned an object label based on learned contextual rules and patch features. A multiscale processing approach ensures locally adapted segmentation levels and improved accuracy.
    5. **Semantic Instance Segmentation:** Labeled patches are merged into individual objects based on learned patch-object relationships and contextual information, resulting in instance-level segmentation of the scene.
