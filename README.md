# k-graph-laplacian
 
 K-graph Laplacians for supervised metric learning
 
 Metric learning is concerned with the generation of adaptive distance functions for each dataset prior to clustering and classification tasks. Most graph-based approaches employ extrinsic distances, such as the Euclidean distance, to weight the edges of the similarity graph built from the samples. In this paper, we propose the definition of K-graphs, using the notion of local curvature from differential geometry to provide an intrinsic cost function for the edges. By approximating the tangent spaces with the PCA subspace, it is possible to compute patch-based estimatives for the principal curvatures of an edge in the graph, in a way that we assign the minimum local curvature to within-class edges and the sum of the maximum local curvatures for between-class edges. The eigenvectors associated to the smallest non-zero aigenvalues of the K-graph Laplacian capture relevant information in terms of data clustering. Experiments with several real datasets show that the proposed method can obtain better clusters than some well known dimensionality reduction based metric learning algorithms such as Supervised PCA, LDA, t-SNE and UMAP.
