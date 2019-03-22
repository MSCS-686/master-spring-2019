from kMeans import KMeans
from sklearn.datasets.samples_generator import make_blobs


X, cluster_assignments = make_blobs(n_samples=200, centers=4,
cluster_std=0.60, random_state=0)

clusters, centroids = KMeans(4,10).fit(X)
print(centroids)
print(clusters)
print(cluster_assignments)

correct = 0
incorrect = 0
for i in range(len(cluster_assignments)):
    if clusters[i] == cluster_assignments[i]:
        correct += 1
    else:
        incorrect += 1

print("correct: {}, Incorrect: {}".format(correct,incorrect))
