import sys
from sklearn.feature_extraction.text import CountVectorizer

class BagOfWords:
    def bag_of_words(self, choice=1):
        # read lines
        raw_data = open('./trn.data').read().strip().split('\n')


        # fit transfrom and get the vectorized data
        if choice == 1:
            vectorizer = CountVectorizer(lowercase=True, min_df=2)
        elif choice == 2:
            vectorizer = CountVectorizer(lowercase=True, max_df=0.5)
        elif choice == 3:
            vectorizer = CountVectorizer(lowercase=True, min_df=2, max_df=0.5)
        elif choice == 4:
            vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=2)
        else:
            sys.stderr.write("Please use valid choice for CountVectorizer!")

        # at this point, transformed_data is considered as a sparse
        transformed_data = vectorizer.fit_transform(raw_data)
        print(transformed_data.shape)




bag = BagOfWords()
bag.bag_of_words(4)

