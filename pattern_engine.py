#This will be where i create the functions to enable easier use of pattern
from pattern.en import sentiment, Sentence, parse, mood, modality

#these two functions assume that the data is coming in as a list of lists, where each list is a set of tweet data, with the tweet_id as the first item, and that we just want to add data to those records, based off of the tweet content, which is in the [2] position. 

def add_sentiment(tdb):
        tweet_db = tdb
        for tweet in tweet_db:
                (polarity, subjectivity) = sentiment(tweet[2])
                tweet.append(polarity)
                tweet.append(subjectivity)
        return tweet_db


def add_modality(tdb):
        for tweet in tdb:
                s = parse(tweet[2], lemmata=True)
                s = Sentence(s)
                (form, score) = (mood(s), modality(s))
                tweet.extend((form, score))
        return tdb
