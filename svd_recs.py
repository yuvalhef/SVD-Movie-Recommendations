import numpy as np
import pandas as pd
import pickle


class RecommenderSystem:

    def __init__(self, train_ratings_mat, test_ratings_mat):

        self.train_ratings_mat = train_ratings_mat
        self.test_ratings_mat = test_ratings_mat
        self.SVD_train_set = []
        self.SVD_valid_set = []
        self.p = []
        self.q = []
        self.bi = []
        self.bu = []
        self.y = []
        self.sum_vecs = []
        self.sqrGroupSize = []
        self.rating_mean = 0
        self.numOfUsers = len(np.unique(train_ratings_mat[:, 0]))
        self.numOfMovies = len(np.unique(train_ratings_mat[:, 1]))
        self.numOfRec = len(train_ratings_mat)
        self.numOfTestRec = len(test_ratings_mat)
        self.prepareData()
        self.type = ""

    def trainImprovedeModel(self, k_latent):
        self.type = "svd_pp"
        # initial parameters
        self.p = np.round((np.random.uniform(-0.05, 0.05, (self.numOfUsers, k_latent))), 3)
        self.q = np.round((np.random.uniform(-0.05, 0.05, (self.numOfMovies, k_latent))), 3)
        self.bu = np.round((np.random.uniform(-0.05, 0.05, self.numOfUsers)), 3)
        self.bi = np.round((np.random.uniform(-0.05, 0.05, self.numOfMovies)), 3)
        self.y = np.round((np.random.uniform(-0.05, 0.05, (self.numOfMovies, k_latent))), 3)
        self.sum_vecs = np.round((np.random.uniform(-0.05, 0.05, (self.numOfUsers, k_latent))), 3)
        self.sqrGroupSize = np.zeros(self.numOfUsers)

        numOfTrainRec = len(self.SVD_train_set)
        numOfValidRec = len(self.SVD_valid_set)

        self.rating_mean = np.mean(self.train_ratings_mat[:, 2])
        lamda = 0.02
        gama = 0.005
        lamda_i = 0.02
        self.SVD_train_set = np.c_[
            self.SVD_train_set, np.zeros(numOfTrainRec), np.zeros(numOfTrainRec)]  # add 2 column2
        self.SVD_valid_set = np.c_[self.SVD_valid_set, np.zeros(numOfValidRec), np.zeros(numOfValidRec)]

        indexArr = []
        movieIndexArr = []
        x = 0
        rmse = 100
        iteration = 0
        print("\n" + "****** Training Model ******")
        while x != -1:
            curr_rmse = rmse
            iteration += 1
            # print for the user to know how far along he/she is with the traning

            print("updating parameters in iteration number: " + str(iteration))
            rec = 0
            for i in range(0, self.numOfUsers):
                Tuser = i + 1
                cnt = 0
                while self.SVD_train_set[rec, 0] == Tuser:
                    cnt += 1
                    rec += 1
                    if rec == numOfTrainRec: break
                indexArr = range(rec - cnt, rec)
                movieIndexArr = np.array(self.SVD_train_set[indexArr, 3] - 1).astype(int)
                self.sqrGroupSize[Tuser - 1] = float(1 / ((cnt) ** 0.5))

                for c in range(rec - cnt, rec):
                    Tmovie = int(self.SVD_train_set[c, 3])
                    # prediction
                    self.sum_vecs[Tuser - 1] = self.y[movieIndexArr].sum(0)
                    self.SVD_train_set[c, 4] = self.predictRating(Tuser, self.SVD_train_set[c, 1])
                    self.SVD_train_set[c, 5] = self.SVD_train_set[c, 2] - self.SVD_train_set[c, 4]
                    err = self.SVD_train_set[c, 5]

                    self.bu[Tuser - 1] += (gama * (err - lamda * self.bu[Tuser - 1]))
                    self.bi[Tmovie - 1] += (gama * (err - lamda * self.bi[Tmovie - 1]))
                    self.p[Tuser - 1] += (gama * (err * self.q[Tmovie - 1] - lamda_i * self.p[Tuser - 1]))

                    self.q[Tmovie - 1] += (gama * (err * (
                                self.p[Tuser - 1] + self.sqrGroupSize[Tuser - 1] * self.sum_vecs[Tuser - 1]) - lamda_i *
                                                   self.q[Tmovie - 1]))
                    b = err * self.sqrGroupSize[Tuser - 1] * self.q[Tmovie - 1]
                    self.y[movieIndexArr] += gama * (b - lamda_i * self.y[movieIndexArr])
                self.sum_vecs[Tuser - 1] = self.y[movieIndexArr].sum(0)
                if i % 1380 == 0:
                    print(str(i / 1380) + "/" + str(int(self.numOfUsers / 1380)))

            print("computing ratings in validation set, iteration number: " + str(iteration))
            for j in range(0, numOfValidRec):
                Vuser = int(self.SVD_valid_set[j, 0])
                self.SVD_valid_set[j, 4] = self.predictRating(Vuser, self.SVD_valid_set[j, 1])
                self.SVD_valid_set[j, 5] = self.SVD_valid_set[j, 2] - self.SVD_valid_set[j, 4]
                if j % 1000000 == 0:
                    print(str(j / 1000000) + "/" + str(int(numOfValidRec / 1000000)))
            rmse = RMSE(self.SVD_valid_set, numOfValidRec)
            print("RMSE for iteration number " + str(iteration) + ": " + str(rmse))
            if rmse >= curr_rmse: x = -1

        print("final RMSE of validation set: " + str(rmse))

    def testImprovedModel(self):

        print("****** Predicting Ratings In Test Set ******")
        n = len(self.test_ratings_mat)
        self.test_ratings_mat = np.c_[self.test_ratings_mat, np.zeros(n), np.zeros(n)]  # add 2 column2

        for j in range(0, n):
            if j % 1000000 == 0:
                # print for the user to know how far along he/she is with the predictions
                print(str(j / 1000000) + "/" + str(int(n / 1000000)))
            user = int(self.test_ratings_mat[j, 0])
            self.test_ratings_mat[j, 4] = self.predictRating(user, self.test_ratings_mat[j, 1])
            self.test_ratings_mat[j, 5] = self.test_ratings_mat[j, 2] - self.test_ratings_mat[j, 4]
        # evaluation
        rmse = RMSE(self.test_ratings_mat, n)
        prec = precision(self.test_ratings_mat, n)
        print("Test Set RMSE: " + str(rmse))
        print("Precision is :" + str(prec))

    def trainBaseModel(self, k_latent):
        self.type = "svd"
        # initialize parameters
        self.p = np.round((np.random.uniform(-0.05, 0.05, (self.numOfUsers, k_latent))), 3)
        self.q = np.round((np.random.uniform(-0.05, 0.05, (self.numOfMovies, k_latent))), 3)
        self.bu = np.round((np.random.uniform(-0.05, 0.05, self.numOfUsers)), 3)
        self.bi = np.round((np.random.uniform(-0.05, 0.05, self.numOfMovies)), 3)

        numOfTrainRec = len(self.SVD_train_set)
        numOfValidRec = len(self.SVD_valid_set)

        self.rating_mean = np.mean(self.train_ratings_mat[:, 2])
        lamda = 0.02
        gama = 0.005

        self.SVD_train_set = np.c_[
            self.SVD_train_set, np.zeros(numOfTrainRec), np.zeros(numOfTrainRec)]  # add 2 column2
        self.SVD_valid_set = np.c_[self.SVD_valid_set, np.zeros(numOfValidRec), np.zeros(numOfValidRec)]

        x = 0

        rmse = 100
        iteration = 0
        print("\n" + "****** Training Model ******")
        while x != -1:
            curr_rmse = rmse
            iteration += 1
            # print for the user to know how far along he/she is with the traning
            print("updating parameters in iteration number: " + str(iteration))
            for i in range(0, numOfTrainRec):
                Tuser = int(self.SVD_train_set[i, 0])
                Tmovie = int(self.SVD_train_set[i, 3])

                self.SVD_train_set[i, 4] = self.predictRating(self.SVD_train_set[i, 0], self.SVD_train_set[i, 1])
                err = self.SVD_train_set[i, 5] = self.SVD_train_set[i, 2] - self.SVD_train_set[i, 4]

                if iteration != 1:
                    self.bu[Tuser - 1] += (gama * (err - lamda * self.bu[Tuser - 1]))
                    self.bi[Tmovie - 1] += (gama * (err - lamda * self.bi[Tmovie - 1]))
                    self.p[Tuser - 1] += (gama * (err * self.q[Tmovie - 1] - lamda * self.p[Tuser - 1]))
                    self.q[Tmovie - 1] += (gama * (err * self.p[Tuser - 1] - lamda * self.q[Tmovie - 1]))

                if i % 1000000 == 0:
                    print(str(i / 1000000) + "/" + str(int(numOfTrainRec / 1000000)))
            print("computing ratings in validation set, iteration number: " + str(iteration))
            for j in range(0, numOfValidRec):
                self.SVD_valid_set[j, 4] = self.predictRating(self.SVD_valid_set[j, 0], self.SVD_valid_set[j, 1])
                self.SVD_valid_set[j, 5] = self.SVD_valid_set[j, 2] - self.SVD_valid_set[j, 4]
                if j % 1000000 == 0:
                    print(str(j / 1000000) + "/" + str(int(numOfValidRec / 1000000)))
            rmse = RMSE(self.SVD_valid_set, numOfValidRec)
            print("RMSE for iteration number " + str(iteration) + ": " + str(rmse))

            if rmse >= curr_rmse: x = -1

        print("final RMSE of validation set: " + str(rmse))


    def testBaseModel(self, ret_pred=False, cos_dict={}):

        print("****** Predicting Ratings In Test Set ******")
        n = len(self.test_ratings_mat)
        self.test_ratings_mat = np.c_[self.test_ratings_mat, np.zeros(n)]
        self.test_ratings_mat = np.c_[self.test_ratings_mat, np.zeros(n)]

        for j in range(0, n):
            if j % 1000000 == 0:
                # print for the user to know how far along he/she is with the predictions
                print(str(j / 1000000) + "/" + str(int(n / 1000000)))

            self.test_ratings_mat[j, 4] = self.predictRating(self.test_ratings_mat[j, 0], self.test_ratings_mat[j, 1],
                                                             cos_dict)
            self.test_ratings_mat[j, 5] = self.test_ratings_mat[j, 2] - self.test_ratings_mat[j, 4]

        if ret_pred == False:
            # evaluate
            rmse = RMSE(self.test_ratings_mat, n)
            prec = precision(self.test_ratings_mat, n)

            print("Test Set RMSE: " + str(rmse))
            print("Precision is :" + str(prec))
        else:
            # return predictions
            return self.test_ratings_mat

    def predictRating(self, userId, movieId, similarities_dict={}):

        uId = int(userId)
        # convert movieId to the appropriate number in dict, to get the correct prediction
        if int(movieId) in self.dict:
            # this is a new movie - does not appear in the test set
            mId = int(self.dict[int(movieId)])

        if uId > self.numOfUsers:
            # this is a new user
            return self.bi[mId - 1] + self.rating_mean

        if self.type == "svd":
            # predict by svd self
            return np.dot(self.p[uId - 1], self.q[mId - 1]) + self.bi[mId - 1] + self.bu[uId - 1] + self.rating_mean

        elif self.type == "svd_pp":
            # predict by svd ++

            return np.dot(self.p[uId - 1] + self.sqrGroupSize[uId - 1] * self.sum_vecs[uId - 1], self.q[mId - 1]) + \
                   self.bi[mId - 1] + self.bu[uId - 1] + self.rating_mean



    def prepareData(self):
        print("****** Preparing Data ******")
        self.train_ratings_mat = self.train_ratings_mat[self.train_ratings_mat[:, 1].argsort()]  # sort by movieId
        self.train_ratings_mat = np.c_[self.train_ratings_mat, np.zeros(self.numOfRec)]
        currI = self.train_ratings_mat[0, 1]
        f = 0
        print("Preparing training set")
        for movieIndex in range(0, self.numOfMovies):
            if movieIndex % 5000 == 0:
                print(str(movieIndex / 5000) + "/" + str(int(self.numOfMovies / 5000)))
            if f < self.numOfRec:
                curr = self.train_ratings_mat[f, 1]
            while (f < self.numOfRec):
                currI = self.train_ratings_mat[f, 1]
                if currI != curr: break
                self.train_ratings_mat[f, 3] = movieIndex + 1
                f += 1
        # prepare test set
        print("Preparing test set")
        self.test_ratings_mat = self.test_ratings_mat[self.test_ratings_mat[:, 1].argsort()]  # sort by movieId
        self.test_ratings_mat = np.c_[self.test_ratings_mat, np.zeros(self.numOfTestRec)]

        a = np.array((np.unique(self.train_ratings_mat[:, 1]), np.unique(self.train_ratings_mat[:, 3])))
        self.dict = dict(zip(a[0], a[1]))
        for i in range(0, self.numOfTestRec):
            if i % 1000000 == 0:
                print(str(i / 1000000) + "/" + str(int(self.numOfTestRec / 1000000)))
            if self.test_ratings_mat[i, 1] in self.dict:
                self.test_ratings_mat[i, 3] = self.dict[self.test_ratings_mat[i, 1]]

        # cut records with untrained movies
        self.test_ratings_mat = self.test_ratings_mat[self.test_ratings_mat[:, 3].argsort()]
        self.test_ratings_mat = self.test_ratings_mat[self.test_ratings_mat[:, 0].argsort()]

        # split to validation and training sets
        indexToMove = np.zeros(int(round(0.23 * self.numOfRec)))
        self.train_ratings_mat = self.train_ratings_mat[self.train_ratings_mat[:, 0].argsort()]

        j = 0
        currIndex = 0
        print("Creating validation set")
        # create validation set, making sure users were in both sets
        for i in range(1, self.numOfUsers + 1):
            if i % 10000 == 0:
                print(str(i / 10000) + "/" + str(int(self.numOfUsers / 10000)))
            cnt = 0
            while (j < self.numOfRec - 1) & (self.train_ratings_mat[j, 0] <= i):
                j += 1
                cnt += 1
            if (cnt == 0):
                continue
            numToMove = int(round(cnt * 0.2))
            indexToMove[currIndex:(currIndex + numToMove)] = (
                np.random.choice(range(j - cnt, j), numToMove, replace=False))
            currIndex = currIndex + numToMove
        indexToMove = indexToMove[:currIndex]

        self.SVD_valid_set = self.train_ratings_mat[indexToMove.astype(int), :]
        ind = np.setdiff1d(range(0, self.numOfRec), indexToMove.astype(int))
        self.SVD_train_set = self.train_ratings_mat[ind, :]

    def getType(self):
        # get model type
        return self.type


def RMSE(ratings_mat, n):
    # RMSE calculation for SVD and SVD++, aggregating the errors
    err_sum = 0
    for k in range(0, n):
        err_sum = err_sum + (ratings_mat[k, 5]) ** 2
    return round((err_sum / n) ** 0.5, 3)


def precision(ratings_mat, n):
    rec = 0
    numOfUsers = len(np.unique(ratings_mat[:,0]))
    precSum = 0
    skip = 0
    for i in range(0, numOfUsers):
        cnt = 0
        while ratings_mat[rec, 0] == i+1:
            cnt += 1
            rec += 1
            if rec == n: break
        indexArr = range(rec - cnt, rec)
        predRel = ratings_mat[indexArr][(ratings_mat[indexArr, 4] >= 3.75), 1]
        realRel = ratings_mat[indexArr][(ratings_mat[indexArr, 2] >= 4.0), 1]
        prec = 0
        frac = 0
        N = len(predRel)
        l = len(realRel)
        if N < 5:
            skip += 1
            continue
        for j in range(0, l):
            frac += np.count_nonzero(predRel == realRel[j])
        prec = float(frac)/float(N)
        precSum += prec


    return round((float(precSum)/float(numOfUsers-skip)), 3)


def precision_recall(preds, trueValues):
    # content based precision calculation: TP/(TP+FP)
    preds.loc[preds <= 4.0] = 0
    preds.loc[preds > 4.0] = 1
    trueValues.loc[trueValues < 4.0] = 0
    trueValues.loc[trueValues >= 4.0] = 1
    sum_pred_true = preds + trueValues
    tp = sum_pred_true[sum_pred_true == 2].count()
    fp = preds[(preds == 1) & (trueValues == 0)].count()
    fn = preds[(preds == 0) & (trueValues == 1)].count()
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    return precision, recall




def Load(path):
    # loading the files and turning it into np.arrays for better speed
    # zf = zipfile.ZipFile(path)
    train_data = pd.read_pickle('C:/Users/user/Desktop/ML/RS/SVD1/trainS.pickle')
    train_mat = train_data[['user_id', 'business_id', 'stars']].values
    test_data = pd.read_pickle('C:/Users/user/Desktop/ML/RS/SVD1/testS.pickle')
    test_mat = test_data[['user_id', 'business_id', 'stars']].values
    del train_data, test_data
    with open('C:/Users/user/Desktop/ML/RS/project/Explainable-Recommendation-master/data/datas_popularity.pickle',
              'rb') as f:  # Python 3: open(..., 'rb')
     items_data, A_test, A_test_dense, A_train, A_train_dense, A_valid, A_valid_dense, test, train, validation, user_index, feature_index, user_dict, product_index, product_dict = pickle.load(f)

    for i in range(len(train_mat)):
        train_mat[i, 0] = user_index[train_mat[i, 0]]
        train_mat[i, 1] = product_index[train_mat[i, 1]]
    for i in range(len(test_mat)):
        test_mat[i, 0] = user_index[test_mat[i, 0]]
        test_mat[i, 1] = product_index[test_mat[i, 1]]
    train_ratings_mat = np.array(train_mat)
    test_ratings_mat = np.array(test_mat)

    del train_mat, test_mat
    return train_ratings_mat, test_ratings_mat


if __name__ == '__main__':

    print("\nhi!")
    # traning menu
    option = input("Choose An Option From The Menu: \n" \
                   "1. Train SVD Model\n" \
                   "2. Train SVD++ Model\n" \
                   "3. Train Content Based Model\n" \
                   "4. Train combined model\n\n")

    while (option != -1):
        # read files and initialize model
        path = input("Please insert the path to the ml20m zip file: ")
        print("Loading Files\n")
        train_ratings_mat, test_ratings_mat = Load(path)
        model = RecommenderSystem(train_ratings_mat, test_ratings_mat)

        if int(option) == 1:
            # train svd
            model.trainBaseModel(30)
            option = -1

        elif int(option) == 2:
            # train svd++
            model.trainImprovedeModel(30)
            option = -1

        else:
            option = input("Please insert a valid option: ")

    option = 0
    print("\nTraining process has finished\n")

    while option != -1:
        # predictions menu
        option = input("\n Please choose an option:\n" \
                       "\n1. Test the model on the test set\n" \
                       "2. Predict a rating specific\n"
                       "99. exit\n")
        if int(option) == 1:
            # give all predictions for the test set
            if model.getType() == "svd":
                # predict with svd
                model.testBaseModel()
            elif model.getType() == "svd_pp":
                # predict with svd++
                model.testImprovedModel()

        elif int(option) == 2:
            # predict for a single combination of userId and movieId
            mId = input("Insert Movie ID: ")
            vId = input("Insert User ID: ")
            if (model.type == "contentBased") or (model.type == "Combined"):
                # send the cosine similarities matrix in case of use of content ased model
                print(model.predictRating(int(vId), int(mId), similarities_dict=cos_sim_dict))
            else:
                print(model.predictRating(int(vId), int(mId)))

        elif int(option) == 99:
            print("Bye!\n")
            option = -1
