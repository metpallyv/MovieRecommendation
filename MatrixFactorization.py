#Implementing regularized Non-negative Matrix factorization using Regularized gradient descent
__author__ = 'vardhaman'
import sys, numpy as np
from numpy import genfromtxt
import codecs
from numpy import linalg as LA

#build movie dicitionary with line no as numpy movie id ,its actual movie id as the key.
def build_movies_dict(movies_file):
    i = 0
    movie_id_dict = {}
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            if i == 0:
                i = i+1
            else:
                movieId,title,genres = line.split(',')
                movie_id_dict[int(movieId)] = i-1
                i = i +1
    return movie_id_dict

#Each line of i/p file represents one tag applied to one movie by one user,
#and has the following format: userId,movieId,tag,timestamp
#make sure you know the number of users and items for your dataset
#return the sparse matrix as a numpy array
def read_data(input_file,movies_dict):
    #no of users
    users = 718
    #users = 5
    #no of movies
    movies = 8927
    #movies = 135887
    X = np.zeros(shape=(users,movies))
    i = 0
    #X = genfromtxt(input_file, delimiter=",",dtype=str)
    with open(input_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                #print "i is",i
                user,movie_id,rating,timestamp = line.split(',')
                #get the movie id for the numpy array consrtruction
                id = movies_dict[int(movie_id)]
                #print "user movie rating",user, movie, rating, i
                X[int(user)-1,id] = float(rating)
                i = i+1
    return X

# non negative regulaized matrix factorization implemention
def matrix_factorization(X,P,Q,K,steps,alpha,beta):
    Q = Q.T
    for step in xrange(steps):
        print step
        #for each user
        for i in xrange(X.shape[0]):
            #for each item
            for j in xrange(X.shape[1]):
                if X[i][j] > 0 :

                    #calculate the error of the element
                    eij = X[i][j] - np.dot(P[i,:],Q[:,j])
                    #second norm of P and Q for regularilization
                    sum_of_norms = 0
                    #for k in xrange(K):
                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
                    #added regularized term to the error
                    sum_of_norms += LA.norm(P) + LA.norm(Q)
                    #print sum_of_norms
                    eij += ((beta/2) * sum_of_norms)
                    #print eij
                    #compute the gradient from the error
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

        #compute total error
        error = 0
        #for each user
        for i in xrange(X.shape[0]):
            #for each item
            for j in xrange(X.shape[1]):
                if X[i][j] > 0:
                    error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
        if error < 0.001:
            break
    return P, Q.T

#main function
def main(X,K):
    #no of users
    N= X.shape[0]
    #no of movies
    M = X.shape[1]
    #P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
    P = np.random.rand(N,K)
    #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
    Q = np.random.rand(M,K)
    #steps : the maximum number of steps to perform the optimisation, hardcoding the values
    #alpha : the learning rate, hardcoding the values
    #beta  : the regularization parameter, hardcoding the values
    steps = 5000
    alpha = 0.0002
    beta = float(0.02)
    estimated_P, estimated_Q = matrix_factorization(X,P,Q,K,steps,alpha,beta)
    #Predicted numpy array of users and movie ratings
    modeled_X = np.dot(estimated_P,estimated_Q.T)
    np.savetxt('mf_result.txt', modeled_X, delimiter=',')

if __name__ == '__main__':
    #MatrixFactorization.py <rating file>  <no of hidden features>  <movie mapping file>
    if len(sys.argv) == 4:
        ratings_file =  sys.argv[1]
        no_of_features = int(sys.argv[2])
        movies_mapping_file = sys.argv[3]

        #build a dictionary of movie id mapping with counter of no of movies
        movies_dict = build_movies_dict(movies_mapping_file)
        #read data and return a numpy array
        numpy_arr = read_data(ratings_file,movies_dict)
        #main function
        main(numpy_arr,no_of_features)
