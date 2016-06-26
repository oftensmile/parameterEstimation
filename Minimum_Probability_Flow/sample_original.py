# Online, local Minimum Probability Flow (MPF) learning rule
# achieves exponential storage from few samples
#
# code supplement for paper:
#    C. Hillar and N. Tran, Robust exponential memory in Hopfield networks, 2015.
#                 arXiv: http://arxiv.org/abs/1411.4625
#    Exp Storage Paper:  http://www.msri.org/people/members/chillar/files/nature_cliquenet.pdf
#    Python code: http://www.msri.org/people/members/chillar/files/local_mpf_rule.txt
#
# See also:
#    C. Hillar, J. Sohl-Dickstein, K. Koepsell, Efficient and optimal binary Little-Hopfield
#               associative memory storage using minimum probability flow, NIPS (DISCML), 2012
#    http://www.msri.org/people/members/chillar/files/mpf_hopfield.pdf
#
# C. Hillar, May, 2015
#
# Note: Tested to work under Python 2
# 
# [LICENSED FOR ACADEMIC, NON-COMMERCIAL USE ONLY]
#


import numpy as np


def ind(i, j, V):
    """ returns index into (1D) internal array edges of V(V-1)/2 binary vector """
    if i < j:
        return (i * V - i*(i+1) // 2 + j - i - 1)
    return (j * V - j * (j + 1) // 2 + i - j - 1)

def sample_clique(V, K):
    """ produce a random K-clique on an V-node graph """
    edges = np.zeros(V * (V-1) // 2, dtype=int)
    clique_array = np.random.permutation(V)[0:K]  # clique vertices
    for i in clique_array:
        for j in clique_array:
            if i != j:
                edges[ind(i, j, V)] = 1
    return edges

def sample_bipartite(V, K, adjacency_matrix=False):
    # left vertices
    left_verts = np.random.permutation(V)[0:K]
    bipart = np.zeros((V, V))
    edges = np.zeros(V * (V-1) // 2, dtype=int)
    for i in left_verts:
        for j in xrange(V):
            if j not in left_verts:
                bipart[i, j] = 1; bipart[j, i] = 1
                edges[ind(i, j, V)] = 1
    if adjacency_matrix:
        return bipart
    return edges

def dynamics_update_single(x, J, theta, max_iter=10 ** 2):
    """ recurrent dynamics (asynchronous version) """
    update = np.asarray((np.sign(np.dot(J, x) - theta) + 1) / 2, dtype=int)
    count = 0
    while (not (update == x).all()) and (count < max_iter):
        temp = update
        update = np.zeros(x.shape)
        for i in xrange(len(x)):
            update[i] = (np.sign(np.dot(J[i], x) - theta[i]) + 1) / 2
        x = temp
        count += 1
    return update

def dynamics_update(X, J, theta, max_iter=10 ** 2):
    X = np.atleast_2d(X)
    Y = np.zeros(X.shape)
    for c, x in enumerate(X):
        Y[c] = dynamics_update_single(x, J, theta, max_iter=max_iter)
    return Y

def mpf_opr_update(X):
    """ e^(x) ~ 1 approximation MPF update rule """
    J = np.zeros((X.shape[1], X.shape[1]))
    theta = np.zeros(X.shape[1])
    X = np.atleast_2d(X)
    for x in X:
        d = (1. - 2. * x)
        J -= np.outer(d, x)
        theta += d
    J[np.eye(X.shape[1], dtype=bool)] *= 0
    return J, theta

def mpf_objective_gradient(X, J, return_obj=False, low=-40, high=40):
    """ J is a square np.array with -2 * theta on diagonal
        X is a M x N np.array of binary vectors 
        NOTE: This is the MPF objective function / gradient
        with 2J (2 theta) replacing J (theta) in the MPF objective:
            Flow = 1/|X| sum_{x in X} sum_{x' bit flip of x} exp(Ex - Ex')
        This gives the same MIN but is easier to manipulte (no 1/2 in the exp)
        Divide parameters by 2 to get ARGMIN of original MPF objective
        (although unnecessary since dynamics is the same)
    """
    X = np.atleast_2d(X)
    M, N = X.shape
    S = 2 * X - 1
    F = -S * np.dot(X, J.T) + .5 * np.diag(J)[None, :]
    Kfull = np.exp(np.clip(F, low, high))  # to avoid exp overflows
    dJ = -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
    dJ = .5 * (dJ + dJ.T)
    if return_obj:
        return Kfull.sum() / M, dJ / M
    else:
        return dJ / M

def mpf_update(X, J, theta):
    """ full MPF local update rule """
    J[np.eye(J.shape[0], dtype=bool)] = -2 * theta
    DJ = mpf_objective_gradient(X, J)
    Dtheta = -.5 * np.diag(DJ)
    DJ[np.eye(J.shape[0], dtype=bool)] *= 0
    J[np.eye(J.shape[0], dtype=bool)] *= 0
    return DJ, Dtheta


#####################
# Begin tests (should take < 30 minutes on modern desktop):
#####################
print 'Storing random patterns with zeroth-order (Hebbian) approximation to MPF learning rule...'
N = 1000; M = 4
X = np.random.random((M, N)) > .7
J, theta = mpf_opr_update(X)
J = (J + J.T) / 2
print "M=%d patterns in %d-node nets: (exact correct | avg bits)" % (M, N), (X == dynamics_update(X, J, theta)).all(1).mean(), (X == dynamics_update(X, J, theta)).mean()

N = 400; M = 200
print 'Storing many big random patterns with online MPF learning rule (%d total parameters to learn)...' % int(((N + 1) * N / 2))
J = np.zeros((N, N)); theta = np.zeros(N)
X = np.random.random((M, N)) > .5
alpha = .01
for j in xrange(1800):
    DJ, DT = mpf_update(X, J, theta)
    J -= alpha * DJ
    theta -= alpha * DT
J = (J + J.T) / 2
print "M=%d patterns in %d-node nets: " % (M, N), (X == dynamics_update(X, J, theta)).all(1).mean(), (X == dynamics_update(X, J, theta)).mean()
# plt.hist(J.ravel(), 100)

print 'Storing 1.3N random patterns with MPF learning rule...'
N = 100; M = 130
J = np.zeros((N, N)); theta = np.zeros(N)
X = np.random.random((M, N)) > .5
alpha = .08
for j in xrange(35000):
    DJ, DT = mpf_update(X, J, theta)
    J -= alpha * DJ
    theta -= alpha * DT
J = (J + J.T) / 2
print "M=%d patterns in %d-node nets: " % (M, N), (X == dynamics_update(X, J, theta)).all(1).mean(), (X == dynamics_update(X, J, theta)).mean()

print 'Trying [but fails] to store 1.8N random patterns with MPF rule (critical # is M ~ 1.6N)...'
N = 100; M = 180
J = np.zeros((N, N)); theta = np.zeros(N)
X = np.random.random((M, N)) > .5
alpha = .1
for j in xrange(17000):
    DJ, DT = mpf_update(X, J, theta)
    J -= alpha * DJ
    theta -= alpha * DT
J = (J + J.T) / 2
print "M=%d patterns in %d-node nets: " % (M, N), (X == dynamics_update(X, J, theta)).all(1).mean(), (X == dynamics_update(X, J, theta)).mean()

V = 32; K = 16; N = V * (V - 1) / 2; S = 600; NCliques = 601080390 # = scipy.misc.comb(32, 16)
print 'Storing cliques with MPF learning rule...(%d total parameters to learn)' % int((N + 1) * N / 2)
X = np.array([sample_clique(V, K) for i in xrange(S)])
J = np.zeros((N, N)); theta = np.zeros(N)
alpha = .02
for j in xrange(4000):
    DJ, DT = mpf_update(X, J, theta)
    J -= alpha * DJ
    theta -= alpha * DT
J = (J + J.T) / 2
novel_cliques = np.array([sample_clique(V, K) for c in xrange(100)])
print "Stored (%d) %d-cliques in V=%d vertex graphs by minimizing prob flow on %d samples (%d-node net): " % (NCliques, K, V, S, N), (novel_cliques == dynamics_update(novel_cliques, J, theta)).all(1).mean(), (novel_cliques == dynamics_update(novel_cliques, J, theta)).mean()

V = 40; K = 15; N = V * (V - 1) / 2; S = 1200; NCliques = 40225345056 # = scipy.misc.comb(40, 15)
print 'Storing cliques with MPF learning rule...(%d total parameters to learn)' % int((N + 1) * N / 2)
X = np.array([sample_clique(V, K) for i in xrange(S)])
J = np.zeros((N, N)); theta = np.zeros(N)
alpha = .02
for j in xrange(8000):
    DJ, DT = mpf_update(X, J, theta)
    J -= alpha * DJ
    theta -= alpha * DT
J = (J + J.T) / 2
novel_cliques = np.array([sample_clique(V, K) for c in xrange(100)])
print "Stored (%d) %d-cliques in V=%d vertex graphs by minimizing prob flow on %d samples (%d-node net): " % (NCliques, K, V, S, N), (novel_cliques == dynamics_update(novel_cliques, J, theta)).all(1).mean(), (novel_cliques == dynamics_update(novel_cliques, J, theta)).mean()

#######################################################
# Run this to see that these networks can store other exponentially large
# combinatorial structures (complete bipartite graphs)
#######################################################
# print '[Extra] Storing complete bipartite graphs with MPF learning rule...'
# V = 30; K = 15; N = V * (V - 1) / 2; S = 8000; NBipartite = 64512240 # = scipy.misc.comb(30, 15)
# X = np.array([sample_bipartite(V, K) for i in xrange(S)])
# J = np.zeros((N, N)); theta = np.zeros(N)
# alpha = .01
# for j in xrange(15000):
#     DJ, DT = mpf_update(X, J, theta)
#     J -= alpha * DJ
#     theta -= alpha * DT
# J = (J + J.T) / 2
# novel_bipartite = np.array([sample_bipartite(V, K) for c in xrange(100)])
# print "Stored (%d) complete (%d, %d)-bipartite graphs in V=%d vertex graphs by minimizing prob flow on %d samples (%d-node net): " % (NBipartite, K, K, V, S, N), (novel_bipartite == dynamics_update(novel_bipartite, J, theta)).all(1).mean(), (novel_bipartite == dynamics_update(novel_bipartite, J, theta)).mean()
