#this python file reads the Predicted Matrix for users and movie ratings and recommends
#50 tops movies for each user based on his/her unrated movies
#I implemented this seperately from building model as once the model is built, we can use it many times
__author__ = 'vardhaman'
import sys, numpy as np
import codecs
import operator
reload(sys)
sys.setdefaultencoding('utf8')

#function to return a dictionary with actual movie id as key and its line no as movie id for numpy array
def dict_with_movie_and_id(movies_file):
    movies_names_dict = {}
    movies_id_dict ={}
    i = 0
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            #print line
            if i == 0:
                i = i+1
            else:
                movie_id, movie_name, genre = line.split(',')
                #print movie_id,movie_name
                movies_names_dict[int(movie_id)] = movie_name
                movies_id_dict[int(movie_id)] = i-1
                i = i +1
    return movies_names_dict, movies_id_dict

#function to return a dictionary with users along with non-rated movie
def dict_with_user_unrated_movies(rating_file,movie_mapping_id):
    #no of users
    users = 718
    #users = 5
    #no of movie ids
    #movies = 4
    movies = 8927
    dict_with_unrated_movies_users ={}
    X = np.zeros(shape=(users,movies))
    i = 0
    with open(rating_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                user,movie,rating,timestamp = line.split(',')
                id = movie_mapping_id[int(movie)]
                #print "user movie rating",user, movie, rating, i
                X[int(user)-1,id] = float(rating)
                i = i+1
    #print X
    for row in xrange(X.shape[0]):
        unrated_movi_ids = np.nonzero(X[row] == 0)
        #print "user",row+1, "has unrated movies", list(unrated_movi_ids[0])
        unrated_movi_ids = list(unrated_movi_ids[0])
        unrated_movi_ids = map(lambda x: x+1,unrated_movi_ids)
        dict_with_unrated_movies_users[row+1] = unrated_movi_ids
    #print "dict with unrated movies",dict_with_unrated_movies_users
    return dict_with_unrated_movies_users

#build predicted numpy array from the comma seperated file
def build_predicted_numpy_array(pred_file):
    #no of users
    users = 718
    #users = 5
    #no of movie ids
    #movies = 4
    movies = 8927
    X = np.zeros(shape=(users,movies))
    user = 0
    with open(pred_file,'r') as f:
        for line in f:
            ratings = line.split(',')
            for movie_id,rating in enumerate(ratings):
                X[user,movie_id] = rating
            user = user+1
    #print "predicted matrix is", X
    return X

#recommend top 25 movies for user specified
def top_25_recommended_movies(pred_rating_file,users,unrated_movies_per_user,movies_mapping_names,movie_mapping_id):
    #dicitonary with numpy movie id as key and actual movie id as value
    reverse_movie_id_mapping = {}
    for key,val in movie_mapping_id.items():
        reverse_movie_id_mapping[val] = key
    #for each user, predict top 25 movies
    for user in users:
        dict_pred_unrated_movies = {}
        unrated_movies = unrated_movies_per_user[int(user)]
        for unrated_movie in unrated_movies:
            dict_pred_unrated_movies[int(unrated_movie)] = pred_rating_file[int(user)-1][int(unrated_movie)-1]
        #recommend top k movies
        SortedMovies = sorted(dict_pred_unrated_movies.iteritems(), key=operator.itemgetter(1), reverse=True)
        print "Top 25 movies recommendation for the user", user
        for i in range(25):
            movie_id, rating = SortedMovies[i]
            actual_movie_id = reverse_movie_id_mapping[movie_id]
            #recommend movies only if the predicted rating is greater than 3.5
            if rating >= 3.5 :
                print ("{} ".format(movie))
            #print ("{} with Movie rating value {}".format(movies_mapping_names[actual_movie_id],rating))
        print "\n"

#main method
def recommend_movies_for_users(orig_rating_file,pred_rating_file,movies_file,users):
    #method to get the mapping between movie names, actual movie id and numpy movie id
    movies_mapping_names,movie_mapping_id = dict_with_movie_and_id(movies_file)
    #build predicted numpy movie id from the saved predicted matrix of user and movie ratings
    predicted_rating_numpy_array = build_predicted_numpy_array(pred_rating_file)
    #dictionary of unrated movies for each user
    dict_with_unrated_movies_users = dict_with_user_unrated_movies(orig_rating_file,movie_mapping_id)
    #method which actually recommends top 25 unrated movies based on their the predicted score
    top_25_recommended_movies(predicted_rating_numpy_array,users,dict_with_unrated_movies_users,movies_mapping_names,movie_mapping_id)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        #read the rating file for the missing
        orig_rating_file = sys.argv[1]
        pred_rating_file = sys.argv[2]
        movies_file = sys.argv[3]
        list_of_users = sys.argv[4]
        with open (list_of_users,'r') as f:
          users = f.readline().split(',')
        recommend_movies_for_users(orig_rating_file,pred_rating_file,movies_file,users)
