import cf
import pmf

NUM_USER = 10916
NUM_MOVIE = 5392

def readParam():
    parameter = {}
    with open("parameter.txt", "r") as myfile:
        for line in myfile:
            val = line.strip().split(":")
            parameter[val[0]] = val[1]
    return parameter

def corpusExplore():
    # table 1
    dict_user = {}
    dict_movie = {}
    count_1 = 0
    count_3 = 0
    count_5 = 0
    avg_rating = 0
    # table 2
    user_num_rating = 0
    user_total_rating = 0
    user_count_1 = 0
    user_count_3 = 0
    user_count_5 = 0
    # table 3
    movie_num_rating = 0
    movie_total_rating = 0
    movie_count_1 = 0
    movie_count_3 = 0
    movie_count_5 = 0
    with open("../data/train.csv", "r") as myfile:
        for line in myfile:
            val = line.strip().split(",")
            user = int(val[1])
            movie = int(val[0])
            rate = int(val[2])
            avg_rating += rate
            if not dict_movie.has_key(movie):
                dict_movie[movie] = 1
            if not dict_user.has_key(user):
                dict_user[user] = 1
            if rate == 1:
                count_1 += 1
            elif rate == 3:
                count_3 += 1
            elif rate == 5:
                count_5 += 1
            if user == 4321:
                user_num_rating += 1
                user_total_rating += rate
                if rate == 1:
                    user_count_1 += 1
                elif rate == 3:
                    user_count_3 += 1
                elif rate == 5:
                    user_count_5 += 1
            if movie == 3:
                movie_num_rating += 1
                movie_total_rating += rate
                if rate == 1:
                    movie_count_1 += 1
                elif rate == 3:
                    movie_count_3 += 1
                elif rate == 5:
                    movie_count_5 += 1
    avg_rating /= 820367.0
    # table 1
    print "number of users: ", len(dict_user.keys())
    print "number of movies: ", len(dict_movie.keys())
    print "number of rate 1: ", count_1
    print "number of rate 3: ", count_3
    print "number of rate 5: ", count_5
    print "average rating: ", avg_rating
    print
    # table 2
    print "number of movies for 4321: ", user_num_rating
    print "number of rate 1 for 4321: ", user_count_1
    print "number of rate 3 for 4321: ", user_count_3
    print "number of rate 5 for 4321: ", user_count_5
    print "average rating for 4321: ", float(user_total_rating)/user_num_rating
    print
    # table 3
    print "number of movies for 3: ", movie_num_rating
    print "number of rate 1 for 3: ", movie_count_1
    print "number of rate 3 for 3: ", movie_count_3
    print "number of rate 5 for 3: ", movie_count_5
    print "average rating for 3: ", float(movie_total_rating)/movie_num_rating



def run():
    parameter = readParam()
    print "Parameter: ", parameter
    alg = parameter.get("algorithm")
    if alg == "memory_based":
        cf.memory_based(parameter)
    elif alg == "model_based":
        cf.model_based(parameter)
    elif alg == "pmf":
        pmf.run(parameter)
    else:
        print "wrong algorithm"

if __name__ == "__main__":
    run()
    # corpusExplore()