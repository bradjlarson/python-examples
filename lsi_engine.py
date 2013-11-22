from __future__ import division
from gensim import corpora, models, similarities
from urllib import unquote_plus
import math
import MySQLdb as mdb
import sys
import cPickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

default_stop_list = set("the of and he our ? had it?s there time out know  one you're what just so get like could people \
, - it's some how but av don't their who when we're would do don?t they me his were she her had its to a in for \
is on that by this with i you it not or be are from at as your all have new more an was we will home can us about \
if page my has".split())



#This assumes a query to a MySQL DB, but this can be changed
def get_data(query, con):
	with con:
		cur = con.cursor(mdb.cursors.DictCursor)
		cur.execute(query)
		return cur.fetchall()

#This function assumes takes escaped text and returns the un-escaped form				
def clean_text(dirty):
	return unquote_plus(dirty)

#This is a generator that yields one DB result at a time	
def prep_data(query, con):
	data = get_data(query, con)
	for row in data:
		#cleaned = clean_text(row[article_text])
		yield clean_text(row["article_text"])			

#This function isn't working yet, but eventually it will remove all words that only appear once from the corpus
def remove_singles(texts):
	return [[word for word in text if word not in tokens_once] for text in texts]

#this converts normal text documents (spaces and the like) to tokens		
def to_texts(docs, stop_list=default_stop_list):
	texts = [[word for word in doc.lower().split() if word not in stop_list] for doc in docs]
	return texts

#this converts a collection of tokens to a bag of words count
def to_dict(texts, filename=False):
	dictnry = corpora.Dictionary(texts)
	if filename:
		dictnry.save('%s.dict' % filename)
	return dictnry

#this converts a collection of tokens to an id mapping based on a dictionary
def to_corpus(dictnry, texts, filename=False):
	corpus = [dictnry.doc2bow(text) for text in texts]
	if filename:
		corpora.MmCorpus.serialize('%s.mm' % filename, corpus)
	return corpus

#this returns a corpus and dictionary based on a query and a connection
#It also provides options for a stop list and the ability to save the dictionary and corpus	
def get_corpus(query, con, stop_list=default_stop_list, filename=False):
	docs = prep_data(query, con)
	texts = to_texts(docs, stop_list)
	dictnry = to_dict(texts, filename)
	return (to_corpus(dictnry, texts, filename), dictnry)

#this returns a TF-IDF model and a TF-IDF transformed corpus, with the option to save to filename	
def build_tfidf(corpus, filename=False):
	tfidf = models.TfidfModel(corpus)
	if filename:
		tfidf.save('%s.tfidf' % filename)
	return (tfidf[corpus], tfidf)

#this returns TF-IDF and LSI models, a LSI transformed corpus and an index for searching, with the option to save to a filename	
def build_lsi(corpus, dictnry, filename=False, n_topics=150):
	(t_corpus, tfidf) = build_tfidf(corpus, filename)
	lsi = models.LsiModel(t_corpus, id2word=dictnry, num_topics=n_topics)
	l_corpus = lsi[t_corpus]
	#index = similarities.MatrixSimilarity(l_corpus)
	index = similarities.Similarity('%s_shard' % filename, l_corpus, n_topics)
	if filename:
		lsi.save('%s.lsi' % filename)
		index.save('%s.index' % filename)
	return (l_corpus, tfidf, lsi, index)

#allows you to build a LSI model from just a query and a MySQL connection 	
def model_lsi(query, con, filename=False, stop_list=default_stop_list, n_topics=150):
	(corpus, dictnry) = get_corpus(query, con, stop_list, filename)
	(l_corpus, tfidf, lsi, index) = build_lsi(corpus, dictnry, filename, n_topics)
	return (l_corpus, tfidf, lsi, index, dictnry)

#returns a list with the n best matches in tuple form, requires you to pass in the objects
def query_lsi(query, con, dictnry, tfidf, lsi, index, stop_list=default_stop_list, num_matches=10):
	data = to_texts(prep_data(query, con), stop_list)
	data_bow = [dictnry.doc2bow(doc) for doc in data]
	data_tfidf = tfidf[data_bow]
	data_lsi = lsi[data_tfidf]
	sims = [top_n(index[doc], num_matches) for doc in data_lsi]
	return sims

#returns a list with the n best matches in tuple form, loads the objects from disk
def query_lsi_stored(query, con, filename, stop_list=default_stop_list, num_matches=10):
	dictnry = corpora.Dictionary.load('%s.dict' % filename)
	tfidf = models.TfidfModel.load('%s.tfidf' % filename)
	lsi = models.LsiModel.load('%s.lsi' % filename)
	index = similarities.Similarity.load('%s.index' % filename)
	data = to_texts(prep_data(query, con), stop_list)
	data_bow = [dictnry.doc2bow(doc) for doc in data]
	data_tfidf = tfidf[data_bow]
	data_lsi = lsi[data_tfidf]
	sims = [top_n(index[doc], num_matches) for doc in data_lsi]
	return sims

#returns the n best matches	
def top_n(query, n):	
	sims = sorted(enumerate(query), key=lambda item: -item[1])
	return sims[:n]

#This is a generator that yields one DB result at a time, with two columns,
# one for the db.id, and the other containing the text
def prep_data_id(query, con):
	data = get_data(query, con)
	for row in data:
		#for now i am hard coding the column names, i'll have to add them as passed in values later
		yield [row["article_id"], clean_text(row["article_text"])]

#this converts normal text documents (spaces and the like) to tokens and retains the id value		
def to_texts_id(docs, stop_list=default_stop_list):
	texts = [[doc[0], [word for word in doc[1].lower().split() if word not in stop_list]] for doc in docs]
	return texts

#this converts a collection of tokens to a bag of words count and retains the id value
def to_dict_id(texts, filename=False):
	dictnry = corpora.Dictionary(text[1] for text in texts)
	dictnry.filter_extremes(5, 0.5)
	if filename:
		dictnry.save('%s.dict' % filename)
	return dictnry

#this converts a collection of tokens to an id mapping based on a dictionary as well as an id-to-index mapping
def to_corpus_id(dictnry, texts, filename=False):
	corpus_id = [[text[0], dictnry.doc2bow(text[1])] for text in texts]
	id_mapping = [doc[0] for doc in corpus_id]
	id_mapping = {v : i for i, v in enumerate(id_mapping)}
	#id_dict = {i : {'article_id' : v[0]} for i, v in enumerate(corpus_id)} #could include this as well, but may as well just calc on the fly: , 'bin_bow' : binary_bow(v[1])
	corpus = [doc[1] for doc in corpus_id]
	if filename:
		corpora.MmCorpus.serialize('%s.mm' % filename, corpus)
		cPickle.dump(id_mapping, open('%s.idmap' % filename, 'wb'))
	return (corpus, id_mapping)
	
#this returns a corpus and dictionary and id-to-index mapping based on a query and a connection
#It also provides options for a stop list and the ability to save the dictionary and corpus	
def get_corpus_id(query, con, stop_list=default_stop_list, filename=False):
	docs = prep_data_id(query, con)
	texts = to_texts_id(docs, stop_list)
	dictnry = to_dict_id(texts, filename)
	(corpus_only, id_mapping) = to_corpus_id(dictnry, texts, filename)
	return (corpus_only, dictnry, id_mapping)
	
#allows you to build a LSI model from just a query and a MySQL connection and then map results back to your DB
def model_lsi_id(query, con, filename=False, stop_list=default_stop_list, n_topics=150):
	(corpus, dictnry, id_mapping, ) = get_corpus_id(query, con, stop_list, filename)
	(l_corpus, tfidf, lsi, index) = build_lsi(corpus, dictnry, filename, n_topics)
	return (l_corpus, corpus, tfidf, lsi, index, dictnry, id_mapping)

#returns a list with the n best matches in tuple form, requires you to pass in the objects
def query_lsi_id(query, con, dictnry, tfidf, lsi, index, id_mapping, stop_list=default_stop_list, num_matches=10):
	docs = prep_data_id(query, con)
	texts = to_texts_id(docs)
	(q_corpus, q_id_mapping) = to_corpus_id(dictnry, texts)
	corpus_tfidf = tfidf[q_corpus]
	corpus_lsi = lsi[corpus_tfidf]
	reverse_mapping = invert_dict(id_mapping)
	reverse_query_mapping = invert_dict(q_id_mapping)
	sims = [top_n(index[doc], num_matches) for doc in corpus_lsi]
	sims_id = {reverse_query_mapping[sims.index(sim)] : [(reverse_mapping[tup[0]], tup[1]) for tup in sim if tup[0] in reverse_mapping] for sim in sims if sims.index(sim) in reverse_query_mapping}
	sims_id = {k : [tup for tup in sims_id[k] if tup[0] <> k] for k in sims_id}
	return (q_corpus, q_id_mapping, sims_id)	
				
#returns a list with the n best matches in tuple form, loads the objects from disk
def query_lsi_stored_id(query, con, filename, stop_list=default_stop_list, num_matches=10):
	dictnry = corpora.Dictionary.load('%s.dict' % filename)
	tfidf = models.TfidfModel.load('%s.tfidf' % filename)
	lsi = models.LsiModel.load('%s.lsi' % filename)
	index = similarities.Similarity.load('%s.index' % filename)
	id_mapping = cPickle.load(open('%s.idmap' % filename, 'rb'))
	docs = prep_data_id(query, con)
	texts = to_texts_id(docs)
	(q_corpus, q_id_mapping) = to_corpus_id(dictnry, texts)
	corpus_tfidf = tfidf[q_corpus]
	corpus_lsi = lsi[corpus_tfidf]
	reverse_mapping = invert_dict(id_mapping)
	reverse_query_mapping = invert_dict(q_id_mapping)
	sims = [top_n(index[doc], num_matches) for doc in corpus_lsi]
	sims_id = {reverse_query_mapping[sims.index(sim)] : [(reverse_mapping[tup[0]], tup[1]) for tup in sim if tup[0] in reverse_mapping] for sim in sims if sims.index(sim) in reverse_query_mapping}
	sims_id = {k : [tup for tup in sims_id[k] if tup[0] <> k] for k in sims_id}
	return (q_corpus, q_id_mapping, sims_id)
	
def bridge_lsi_nb(sims):
	sql_stmts = []	
	for sim in sims:
		in_stmt = reduce(lambda x, y: x + str(y[0]) + ", ", sims[sim], "")
		in_stmt = in_stmt[:-2]
		sql = "select article_id, like_flag from unique_likes where article_id in(%s)" % in_stmt
		sql_stmts.append([sim, sql])
	 	#models.append(build_nb(sql, con, id_mapping, corpus))
	return sql_stmts
	
def get_nb_probs(sql_stmts, con, id_mapping, corpus, q_id_mapping, q_corpus, num_tokens=False):
	nb_models = [[stmt[0], build_nb(stmt[1], con, id_mapping, corpus)] for stmt in sql_stmts]
	add_bow = [[model[0], model[1], article_to_bow([model[0]], q_id_mapping, q_corpus)] for model in nb_models]
	probs = [[article[0], nb_classify(article[1], article[2], num_tokens)] for article in add_bow]
	return probs	

def classifier(con, sims, id_mapping, corpus, q_id_mapping, q_corpus, num_tokens=False):
	sql = bridge_lsi_nb(sims)
	probs = get_nb_probs(sql, con, id_mapping, corpus, q_id_mapping, q_corpus, num_tokens)
	return probs
	
def save_results(con, probs, message):
	#cPickle.dump(probs, open('%s.probs' % message, 'wb'))
	with con:
		cur = con.cursor(mdb.cursors.DictCursor)
		sql = "insert into jobs.testing_preds (article_id, message, prediction) values (%s, %s, %s);"
		params = [(prob[0], message, prob[1]) for prob in probs]
		cur.executemany(sql, params)
		con.commit()
	
def get_results(tokens, matches, query, con, filename, id_mapping, o_corpus):
	(q_corpus, q_id_mapping, sims_id) = query_lsi_stored_id(query, con, filename, num_matches=matches)
	probs = classifier(con, sims_id, id_mapping, o_corpus, q_id_mapping, q_corpus, tokens)
	save_results(con, probs, 'num matches=%s, pg bayes=%s' % (matches, tokens))

def run_multiple(num_tokens, num_matches, query, con, filename, id_mapping, o_corpus):
	for token in num_tokens:
		for match in num_matches:
			get_results(token, match, query, con, filename, id_mapping, o_corpus)	
	
#thought is to implement a dictionary to store the id_mappings, with the corpus number as the index
#would then have another dictionary as the value, with keys for any number of values
#each of those values could then be predicted against

#after i get the similar articles, i need to pull the like, dislike information
#step 1: get similar articles from dict
#step 2: split into like list, dislike list
#step 3: map list to word counts from corpus
#step 4: map word counts to binary has word/doesn't
#step 5: reduces the list to a single sparse vector, summing by word across articles
#step 6: reduce that list to a % of category
#step 7: for each word in the text to be classified, map to (% of like / % of like + % of dislike)
#step 8: reduce list by summing (ln(1-p) - ln(p)) across items
#step 9: return prob as (1 / 1 + e^(result from step 8))

#this takes a dictionary and reverses the keys and values
def invert_dict(dictnry):
	return {dictnry[i]: i for i in dictnry}

#this takes a bag of words list, and sets the tuple to either the word is present in the doc (1) or not (0)
def to_bin_bows(bows):
	return [binary_bow(bow) for bow in bows]

#this does the actual conversion to binary bag of word
def binary_bow(b_o_w):
	return [(w[0], one_or_zero(w[1])) for w in b_o_w]

#this converts a number to either one or zero - this should almost always be 1
def one_or_zero(num):
	if num >= 1:
		return 1
	else:
		return 0

#this takes a dictionary, and splits out the article_ids into two lists, one for likes and the other for dislikes
def split_by_like(docs):
	likes = [doc['article_id'] for doc in docs if doc['like_flag'] == 1]
	dislikes = [doc['article_id'] for doc in docs if doc['like_flag'] == 0]
	return (likes, dislikes)

#for these functions, i've transformed them to work on a single article_id at a time.
#I may need to go back and convert them to iterators, we'll see

#the returns a list of the corpus id's (indices) for a given article_id	
def to_index_id(article_ids, id_mapping):
	return [id_mapping[article_id] for article_id in article_ids]

#this takes a list of corpus id's (indices) and returns a list of the bag of words for those id's
def id_to_bow(index_ids, corpus):		
	return [corpus[index_id] for index_id in index_ids]

#this returns a list of bag of words that correspond to a set of article_id's	
def article_to_bow(articles, id_mapping, corpus):
	index_id = to_index_id(articles, id_mapping)
	bow = id_to_bow(index_id, corpus)
	return bow

#this returns a set of like and dislike bag of word lists based on a query
def nb_get_bow(query, con, id_mapping, corpus):
	articles = get_data(query, con)
	(likes, dislikes) = split_by_like(articles)
	like_bows = article_to_bow(likes, id_mapping, corpus)
	dislike_bows = 	article_to_bow(dislikes, id_mapping, corpus)
	bin_like_bows = to_bin_bows(like_bows)
	bin_dislike_bows = to_bin_bows(dislike_bows)
	return (bin_like_bows, bin_dislike_bows)

#this returns a dictionary with % of documents that a word appears in
def word_percent_dict(bin_bows):
	num_articles = len(bin_bows)
	word_dict = {}
	for bow in bin_bows:
		for w in bow:
			if w[0] in word_dict:
				word_dict[w[0]] += 1
			else:
				word_dict[w[0]] = 1
	p_word_dict = {i: get_word_percent(word_dict[i], num_articles) for i in word_dict}
	return p_word_dict
	
#this returns a percentage if it meets some thresholds, otherwise 0.5					
def get_word_percent(num, total, minm=5, default=0.5):
	if num >= minm:
		return (num / total)
	else:
		return default

#this returns the probabiliy of a "yes", given two occurence rates
def get_word_prob(yes_percent, no_percent):
	return (yes_percent / (yes_percent + no_percent)) 		

#this checks if a key is in a dictionary, if it is, returns the value, otherwise returns a default value
def default_to(key, dictnry, default):
	if key in dictnry:
		return dictnry[key]
	else:
		return default

#this combines two sets of word occurences rates, and generates a probabilty of "yes" for each word							
def bayes_prob_dict(yes_probs, no_probs):
	prob_dict = {word: get_word_prob(default_to(word, yes_probs, 0.01), default_to(word, no_probs, 0.01)) for word in set(yes_probs.keys() + no_probs.keys())}
	return prob_dict

#this builds a probability dictionary given a set of articles							
def build_nb(query, con, id_mapping, corpus):
	(likes, dislikes) = nb_get_bow(query, con, id_mapping, corpus)
	return bayes_prob_dict(word_percent_dict(likes), word_percent_dict(dislikes))	

def cutoffs(num, upper=15, lower=-15):
	if num > upper:
		return upper
	elif num < lower:
		return lower
	else:
		return num

#this returns a probability that a bag of words is a "yes"	
def nb_classify(bayes, b_o_w, n):
	probs = [bayes[w[0]] for w in b_o_w[0] if w[0] in bayes]
	probs = sorted(probs)
	if n:
		print n
		probs = probs[:n] + probs[n:]
	print probs	
	nu = reduce(lambda x, y: x + ln_p(y), probs, 0)
	nu = cutoffs(nu)
	return (1 / (1 + math.exp(nu)))
	
#this is used when combining the individual word probabilities, in lieu of multiplying them all together, which avoids issues with float	
def ln_p(prob):	
	return (math.log(1- prob) - math.log(prob))
	

	
		
	
		
	
	


