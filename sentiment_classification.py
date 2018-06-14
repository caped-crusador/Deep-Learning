from __future__ import print_function,division
import random
import tensorflow as tf
import numpy as np
import os

data = []
for file_,label in zip(["class_neg.txt","class_pos.txt"],[0,1]):
    lines = open(file_).readlines()
    lines = list(map(lambda x:x.strip().replace("-"," ").split(),lines))
    for line in lines:
        data.append([line,label])
    print("Number of reviews of {} = {}".format(file_[:-4],len(lines)))
    print("\tMax number of tokens in a sentence = {}".format(max(map(lambda x:len(x),lines))))
    print("\tMin number of tokens in a sentence = {}".format(min(map(lambda x:len(x),lines))))
random.Random(5).shuffle(data)

# See some randomly sampled sentences
print(" ".join(data[random.randint(0,len(data))][0]))

sents = map(lambda x:x[0],data) # all sentences
all_words = set()
for sent in sents:
    all_words |= set(sent)
all_words = sorted(list(all_words))
vocab = {all_words[i]:i for i in range(len(all_words))}
print("Number of words : ",len(vocab))
train = data[:int(0.8*len(data))]
test = data[int(0.8*len(data)):]
train_data = []
train_targets = []
test_data = []
test_targets = []
for list_all,list_data,list_target,label_list in zip([train,test],[train_data,test_data],[train_targets,test_targets],["train","test"]):
    for datum,label in list_all:
        list_data.append([vocab[w] for w in datum])
        list_target.append([label])
    print(label_list)
    print("\tNumber of positive examples : ",list_target.count([1]))
    print("\tNumber of negative examples : ",list_target.count([0]))


def k_max_pool(A, k):
    """
    A = 2 dimensional array (assume that the length of last dimension of A will be always more than k)
    k = number of elements.
    Return: For every row of A, top k elements in the order they appear.
    """
    assert len(A.get_shape()) == 2

    def func(row):
        """
        Hint : I used top_k and reverse.
        I am not sure whether the order of the indices are retained when sorted = False in top_k. (did not find any documentation)
        Therefore, I suggest that you sort the indices before selecting the elements from the array(Trick: use top_k again!)"""
        ret_tensor = None
        ## your code here to compute ret_tensor ##
        index = tf.nn.top_k(row, k=k).indices
        index = tf.contrib.framework.sort(index, axis=-1, direction='ASCENDING', name=None)
        ret_tensor = tf.gather(row, index)

        return ret_tensor

    return tf.map_fn(func, A)

#
# A = tf.placeholder(shape=[None,None],dtype=tf.float64)
# top = k_max_pool(A,5)
# sess = tf.Session()
# for i in range(1,6):
#     np.random.seed(5)
#     l = np.random.randn(i*10,i*10)
#     top_elements = sess.run(top,feed_dict={A:l})
#     l = l.tolist()
#     top_elements2 = np.array(map(lambda x: [x[i] for i in range(len(x)) if x[i]>sorted(x,reverse=True)[5]],l))
#     # Note that this test assumes that the 6th largest element and 5th largest element are different.
#     print(((top_elements - top_elements2)<10**-10).all())


def initializer(shape):
    xavier = tf.contrib.layers.xavier_initializer(seed=1)
    return xavier(shape)


class CNN:
    def __init__(self, num_words, embedding_size=30):
        self.num_words = num_words

        # The batch of text documents. Let's assume that it is always padded to length 100.
        # We could use [None,None], but we'll use [None,100] for simplicity.
        self.input = tf.placeholder(shape=[None, 100], dtype=tf.int32)
        self.expected_output = tf.placeholder(shape=[None, 1], dtype=tf.int32)

        embedding_matrix = tf.Variable(initializer((num_words, embedding_size)), name="embeddings")
        # Add an additional row of zeros to denote padded words.
        v2 = tf.Variable(tf.zeros([1, embedding_size]), dtype=tf.float32)

        self.embedding_matrix = tf.concat([embedding_matrix, 0 * v2], 0)

        # Extract the vectors from the embedding matrix. The dimensions should be None x 100 x embedding_size.
        # Use embedding lookup
        vectors = tf.nn.embedding_lookup(self.embedding_matrix, self.input)  # None x 100 x embedding_size

        # In order to use conv2d, we need vectors to be 4 dimensional.
        # The convention is NHWC - None (Batch Size) x Height(Height of image) x Width(Width of image) x Channel(Depth - similar to RGB).
        # For text, let's consider Height = 1, width = number of words, channel = embedding_size.
        # Use expand-dims to modify.
        vectors2d = tf.expand_dims(vectors, 1)  # None x 1 x 100 x embedding_size

        # Create 50 filters with span of 3 words. You need 1 bias for each filter.
        filter_tri = tf.Variable(initializer((1, 3, embedding_size, 50)), name="weight3")
        bias_tri = tf.Variable(tf.zeros((1, 50)), name="bias3")
        conv1 = tf.nn.conv2d(
            input=vectors2d,
            filter=filter_tri,
            strides=[1, 1, 1, 1],
            padding="VALID"
        )  # Shape = (None x 1 x 98 x 50)
        A1 = tf.nn.leaky_relu(conv1 + bias_tri)

        # Create 50 filters with span of 4 words. You need 1 bias for each filter.
        filter_4 = tf.Variable(initializer((1, 4, embedding_size, 50)), name="weight4")
        bias_4 = tf.Variable(tf.zeros((1, 50)), name="bias4")
        conv2 = tf.nn.conv2d(
            input=vectors2d,
            filter=filter_4,
            strides=[1, 1, 1, 1],
            padding="VALID"
        )  # Shape = ?

        A2 = tf.nn.leaky_relu(conv2 + bias_4)

        # Create 50 filters with span of 5 words. You need 1 bias for each filter.
        filter_5 = tf.Variable(initializer((1, 5, embedding_size, 50)), name="weight5")
        bias_5 = tf.Variable(tf.zeros((1, 50)), name="bias5")
        conv3 = tf.nn.conv2d(
            input=vectors2d,
            filter=filter_5,
            strides=[1, 1, 1, 1],
            padding="VALID"
        )  # Shape = ?

        A3 = tf.nn.leaky_relu(conv3 + bias_5)

        # Now extract the maximum activations for each of the filters. The shapes are listed alongside.
        max_A1 = tf.squeeze(tf.nn.max_pool(A1, ksize=[1, 1, 98, 1], strides=[1, 1, 1, 1], padding='VALID'),
                            [1, 2])  # None x 50
        max_A2 = tf.squeeze(tf.nn.max_pool(A2, ksize=[1, 1, 97, 1], strides=[1, 1, 1, 1], padding='VALID'),
                            [1, 2])  # None x 50
        max_A3 = tf.squeeze(tf.nn.max_pool(A3, ksize=[1, 1, 96, 1], strides=[1, 1, 1, 1], padding='VALID'),
                            [1, 2])  # None x 50

        concat = tf.concat([max_A1, max_A2, max_A3], axis=1)  # None x 150

        # Initialize the weight and bias needed for softmax classifier.
        self.softmax_weight = tf.Variable(initializer((150, 2)), name="W", dtype=tf.float32)
        self.softmax_bias = tf.Variable(tf.zeros(shape=[2]), name="b", dtype=tf.float32)
        logits = tf.matmul(concat, self.softmax_weight) + self.softmax_bias
        # Write out the equation for computing the logits.
        self.output = tf.nn.softmax(logits, axis=1)  # Shape = ?

        # Compute the cross-entropy cost.
        # You might either sum or take mean of all the costs across all the examples.
        # It is your choice as the test case is on Stochastic Training.
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.expected_output, 2), logits=logits)
        self.cost = tf.reduce_mean(entropy)

        correct_prediction = tf.equal(tf.reshape(tf.argmax(self.output, 1), [-1, 1]),
                                      tf.cast(self.expected_output, dtype=tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.cost)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def pad(self, data, pad_word, pad_length=100):
        for datum in data:
            datum.extend([pad_word] * (pad_length - len(datum)))
        return data

    def train(self, train_data, test_data, train_targets, test_targets, batch_size=1, epochs=1, verbose=False):
        sess = self.session
        self.pad(train_data, self.num_words)
        self.pad(test_data, self.num_words)
        print("Starting training...")

        print(sess.run(self.get_distance(["good", "good"], ["bad", "worst"])))
        print(sess.run(self.get_most_similar_word("good")))
        exit()
        for epoch in range(epochs):
            cost_epoch = 0
            c = 0
            for datum, target in zip([train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)],
                                     [train_targets[i:i + batch_size] for i in
                                      range(0, len(train_targets), batch_size)]):
                _, cost = sess.run([self.train_op, self.cost],
                                   feed_dict={self.input: datum, self.expected_output: target})
                cost_epoch += cost
                c += 1
                if c % 100 == 0 and verbose:
                    print("\t{} batches finished. Cost : {}".format(c, cost_epoch / c))
            print("Epoch {}: {}".format(epoch, cost_epoch / len(train_data)))
            print("\tTrain accuracy: {}".format(self.compute_accuracy(train_data, train_targets)))
            print("\tTest accuracy: {}".format(self.compute_accuracy(test_data, test_targets)))
            self.get_most_similar_word("good")

    def get_distance(self, word1, word2):
        indexes1 = [vocab[word1[0]]] * len(word1)
        indexes2 = np.array([vocab[i] for i in word2])
        dist1 = [self.embedding_matrix[indexes1[0]]] * len(indexes1)
        dist2 = [self.embedding_matrix[i] for i in indexes2]
        print("indexes1 : ", indexes1)
        print("indexes2 : ", indexes2)
        print("dist1 : ", dist1)
        print("dist2 : ", dist2)
        cos_similarity = tf.losses.cosine_distance(dist1, dist2, axis=1, reduction=tf.losses.Reduction.NONE)
        # k = np.reciprocal(np.linalg.norm(dist1[0])*(np.linalg.norm(dist2, axis=1)))
        # cos_similarity=np.reshape(np.matmul(dist1[0], np.transpose(dist2)), [1, len(k)])*np.reshape(k,[1,len(k)])
        print (cos_similarity)
        return (cos_similarity)

    #         indexes = np.array([[vocab[i], vocab[j]] for i, j in zip(word1, word2)])
    #         dist = [[self.embedding_matrix[i], self.embedding_matrix[j]] for i, j in zip(indexes[:, 0], indexes[:, 1])]
    #         cos_similarity = tf.losses.cosine_distance(dist[0][0],dist[0][1],axis=0)
    #         print (self.session.run(cos_similarity))
    #         exit(0)
    # return 1-cos_similarity

    def get_most_similar_word(self, word):
        elems = [i for i in vocab.keys() if len(i) > 3]
        word = [word] * len(elems)
        cos_sim = np.array(self.get_distance(word, elems))
        z = (cos_sim[0].argsort()[::-1][:10])
        for i in z:
            print(elems[i])
        return z

    #         embd_matrix = self.session.run(self.embedding_matrix)
    #         elems = tf.convert_to_tensor(list(vocab.keys())[0:50]))
    #         cos_distances = tf.map_fn(lambda x: self.get_distance(x), elems, dtype=tf.float32)
    #         elems = [i for i in vocab.keys() if len(i) > 3]
    #         cos_distances = map(lambda x: self.get_distance(word, x), list(vocab.keys()))
    # #         cos_dist = list(self.get_distance(word, x) for x in list(vocab.keys()))
    # #         cos_dist = tf.convert_to_tensor(cos_distances)
    #         sorted_dist = tf.contrib.framework.argsort(cos_distances)
    #         return sorted_dist[0:10]

    def compute_accuracy(self, data, targets):
        return self.session.run(self.accuracy, feed_dict={self.input: data, self.expected_output: targets})


c=CNN(len(vocab))
c.train(train_data,test_data,train_targets,test_targets,epochs=1,verbose=True)

a = zip("a", [1,2,3])