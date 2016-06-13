import numpy as np
import scipy.sparse as sps
import time

NUM_USER = 10916
NUM_MOVIE = 5392
EPS = 0.5655


# PMF with Gradient Descent
def pmf_GD (parameter):
    '''
    R: N*M rating by user i for movie j
    I: N*M I(i,j) = 1 means there is rating for R(i,j)
    U: D*N latent user matrix
    V: D*M latent movie matrix

    :return: U, V
    '''
    D = int(parameter.get("D"))  # latent factor
    maxIter = 1000
    lamda = 0.0001  # regularization factor
    epsilon = 0.0001  # learning rate
    momentum = 0.4

    # R = np.matrix([[1, 2, 3, 4], [0, 3, 2, 1], [5, 4, 3, 2]])
    # I = np.matrix([[1,1,1,1], [0,1,1,1], [1,1,1,1]])
    tmp_R = sps.lil_matrix((NUM_USER, NUM_MOVIE))
    tmp_I = sps.lil_matrix((NUM_USER, NUM_MOVIE))
    with open("../data/train.csv", "r") as myfile:
        for line in myfile:
            val = line.strip().split(",")
            tmp_R[int(val[1]), int(val[0])] = float(val[2])
            tmp_I[int(val[1]), int(val[0])] = 1
    R = sps.csr_matrix(tmp_R)
    I = sps.csr_matrix(tmp_I)

    N, M = R.shape
    U = np.random.rand(D, N) * 0.05
    V = np.random.rand(D, M) * 0.05
    U = np.matrix(U)
    V = np.matrix(V)
    prev_U_inc = np.zeros(U.shape)
    prev_V_inc = np.zeros(V.shape)

    start = time.time()
    for iter in xrange(maxIter):
        e = R - np.dot(U.transpose(), V)
        e = sps.csr_matrix(e)
        e = e.multiply(I)
        error = e.multiply(e).sum()
        error /= 820367.0
        if error < EPS:
            break
        print error
        # Gradient of V
        sparse_U = sps.csr_matrix(U)
        dV = -sparse_U.dot(e)+lamda * V
        print "Calculate dV"
        # Gradient of U
        sparse_V = sps.csr_matrix(V)
        dU = -sparse_V.dot(e.transpose())+ lamda * U
        print "Calculate dU"
        # Update U and V
        U_inc = momentum * prev_U_inc + epsilon * dU
        V_inc = momentum * prev_V_inc + epsilon * dV
        U = U - U_inc
        V = V - V_inc
        prev_U_inc = U_inc
        prev_V_inc = V_inc
        #epsilon *= 0.995

    duration = time.time() - start
    print "number of iters:", iter
    print "Running Time: ", duration
    return U, V

# PMF with Stochastic Gradient Descent
def pmf_SGD (parameter):
    '''
    R: N*M rating by user i for movie j
    I: N*M I(i,j) = 1 means there is rating for R(i,j)
    U: N*D latent user matrix
    V: M*D latent movie matrix

    :return: U, V
    '''

    D = int(parameter.get("D"))  # latent factor
    maxIter = 100
    lamda = 0.0001  # regularization factor
    epsilon = 0.0002  # learning rate
    num_batch = 200
    momentum = 0.5

    # Train vector is a list of rating pairs {movie_id, user_id, rating}
    # train_vec = [[0,0,1],[0,1,2],[0,2,3],[0,3,4],[1,1,3],[1,2,2],[1,3,1],[2,0,5],[2,1,4],[2,2,3],[2,3,2]]
    # train_vec = np.matrix(train_vec)
    train_vec = []
    with open("../data/train.csv", "r") as myfile:
        for line in myfile:
            val = line.strip().split(",")
            train_vec.append([int(val[0]), int(val[1]), int(val[2])])
    train_vec = np.matrix(train_vec)
    N = NUM_USER     # number of users
    M = NUM_MOVIE     # number of movies
    U = np.random.rand(N, D) * 0.05
    V = np.random.rand(M, D) * 0.05
    prev_U_inc = np.zeros(U.shape)
    prev_V_inc = np.zeros(V.shape)

    time_start = time.time()
    for iter in xrange(maxIter):
        error = 0
        batch_size = train_vec.shape[0]/num_batch
        for batch in xrange(num_batch):
            start = batch*batch_size
            end = (batch+1)*batch_size if batch < num_batch-1 else train_vec.shape[0]
            batch_size = batch_size if batch < num_batch-1 else train_vec.shape[0]-batch*batch_size
            user = train_vec[start:end,1]
            movie = train_vec[start:end,0]
            rating = train_vec[start:end,2]
            U_batch = np.squeeze(U[user,:], 1)
            V_batch = np.squeeze(V[movie,:], 1)
            pred = np.sum(np.multiply(U_batch, V_batch), 1)
            pred = pred.reshape(rating.shape)
            error += np.sum(np.power(rating-pred,2))
            # Compute gradient
            I_out = np.tile(rating - pred, D)
            # print I_out.shape
            I_V = -np.multiply(I_out, U_batch) + lamda * V_batch
            I_U = -np.multiply(I_out, V_batch) + lamda * U_batch
            dU = np.zeros(U.shape)
            dV = np.zeros(V.shape)
            for i in xrange(batch_size):
                dU[user[i,0],:] = dU[user[i,0],:] + I_U[i,:]
                dV[movie[i,0],:] = dV[movie[i,0],:] + I_V[i,:]

            U_inc = momentum * prev_U_inc + epsilon * dU
            V_inc = momentum * prev_V_inc + epsilon * dV
            U = U - U_inc
            V = V - V_inc
            prev_U_inc = U_inc
            prev_V_inc = V_inc
        error /= 820367.0
        if error < EPS:
            break
        print error
    duration = time.time() - time_start
    print "Running Time: ", duration
    # return U.transpose, V.transpose
    return U.transpose(), V.transpose()

def predict(U,V):
    #U = np.matrix(U)
    #V = np.matrix(V)
    print U.shape
    dev_vec = []
    with open("../data/dev.csv", "r") as myfile:
        for line in myfile:
            val = line.strip().split(",")
            dev_vec.append([int(val[0]), int(val[1])])
    test_vec = []
    with open("../data/test.csv", "r") as myfile:
        for line in myfile:
            val = line.strip().split(",")
            test_vec.append([int(val[0]), int(val[1])])
    with open("../data/dev_PMF_pred.csv", "a") as myfile:
        for pair in dev_vec:
            pred = np.dot(U[:,pair[1]].transpose(), V[:, pair[0]])
            myfile.write(str(float(pred)) + "\n")
    with open("../data/test_PMF_pred.csv", "a") as myfile:
        for pair in test_vec:
            pred = np.dot(U[:,pair[1]].transpose(), V[:, pair[0]])
            myfile.write(str(float(pred)) + "\n")

def run(parameter):
    [U, V] = pmf_GD(parameter)
    predict(U, V)

# if __name__ == "__main__":
#     [U, V] = pmf_GD()
#     predict(U, V)