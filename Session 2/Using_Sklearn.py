import timeit
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
import numpy as np

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index, tfidf = int(index_tfidf.split(':')[0]), float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)
    return np.array(data), np.array(labels)

def clustering_with_KMeans():
    data, labels = load_data('../datasets/20news-bydate/20newsgroups_data_tf_idf_full.txt')
    X = csr_matrix(data)
    print('=========')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2018
    ).fit(X)
    clustered_labels = np.array(kmeans.labels_)

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float))/expected_y.size
    return accuracy

def classifying_with_linear_SVMs():
    train_X, train_Y = load_data('../datasets/20news-bydate/20newsgroups_data_tf_idf_train.txt')
    classifier = LinearSVC(
        C = 10.0,
        tol = 0.001,
        verbose = True
    )
    classifier.fit(train_X, train_Y)
    test_X, test_Y = load_data('../datasets/20news-bydate/20newsgroups_data_tf_idf_test.txt')
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_Y, expected_y = test_Y)
    print("Accuracy:", accuracy)

def classifying_with_kernel_SVMs():
    train_X, train_Y = load_data('../datasets/20news-bydate/20newsgroups_data_tf_idf_train.txt')
    classifier = SVC(
        C=50.0,
        kernel = 'rbf',
        gamma = 0.1,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_Y)
    test_X, test_Y = load_data('../datasets/20news-bydate/20newsgroups_data_tf_idf_test.txt')
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_Y, expected_y=test_Y)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    t0 = timeit.default_timer()
    classifying_with_linear_SVMs()
    print("time:", timeit.default_timer() - t0)