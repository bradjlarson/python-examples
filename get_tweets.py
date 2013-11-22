from pattern.web import Twitter
from pattern.en import Sentence, parse, pprint, modality, sentiment, parsetree, ngrams, mood
import db
import MySQLdb as mdb
import sys
import logging
import pattern_engine as p
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

t = Twitter()

trending = t.trends(cached=False)

tweet_db = []

for trend in trending:
	for tweet in t.search(trend, count=10):
		if tweet.language == u'en':
			tweet_db.append((tweet, trend))

tweet_db = [[tweet.id, tweet.author, tweet.text, tweet.date, tweet.profile, tweet.url, tweet.language, trend] for (tweet, trend) in tweet_db]
tweet_db = p.add_sentiment(tweet_db)
tweet_db = p.add_modality(tweet_db)

con = db.con
cur = con.cursor(mdb.cursors.DictCursor)
query = 'INSERT into predictive.dev_pattern \
	(tweet_id, author, text, date, profile, url, language, trend, polarity, subjectivity, mood, modality) \
	VALUES %s, %s, %s, %s, %s, %s, %s, %s, %f, %f, %s, %f);'
for tweet in tweet_db:
	cur.execute(query, tuple(tweet))
	print cur._last_executed
	con.commit()
		
