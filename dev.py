import lsi_engine as _
import db
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#The code here takes some incoming articles from the HN RSS feed,
#and queries them against the set of all articles I've read and classified.
#The model returns the N most similar articles, which it then uses to 
#construct a naive bayesian classifier to predict whether or not I'll like the article
#This way, each naive bayesian classifier is unique to the article, in the theory 
#that the classifier will only contain the most relevant data points. 

#set up DB connection
con = db.con;
#this is the name that our model and data will be saved as
filename = 'testing'
#This is the query we will use to build the model
query = "select a.article_id, article_text from jobs.testing_corpus a, jobs.unique_likes b where a.article_id = b.article_id"
#this returns an lsi transformed corpus, the original corpus, the TF-IDF and LSI models, index, dictnry, and article_id to index id dictionary
(l_corpus, o_corpus, tfidf, lsi, index, dictnry, id_mapping) = _.model_lsi_id(query, con, filename)
#Let's say we want to run the model a bunch of times for a bunch of different parameter values
#Like we want to test the accuracy of using between 50 and 250 results for our bayesian model (num_matches)
#Or, we want to vary how many words are used in the bayesian score from 20 to 200
#So we set those params:
num_matches = [i * 25 for i in range(2,10)]
num_tokens = [i *20 for i in range(1,10)]
#and then run all the combinations of the params, and then we can easily analyze the results to see which combinations are best
_.run_multiple(num_tokens, num_matches, query, con, filename, id_mapping, o_corpus)

#Or we could query the model from above with the data from a new query
#Here we show the steps involved in run_multiple
#first we feed in a query, and the name of the models we want to query against
#which loads the stored models, and returns a converted corpus, and the mappings to match the vectors back to the articles we've liked 
(q_corpus, q_id_mapping, sims_id) = _.query_lsi_stored_id(query, con, filename, num_matches=75)
#then we generate the probability that we'll like the article, based on the most similar articles
probs = _.classifier(con, sims_id, id_mapping, o_corpus, q_id_mapping, q_corpus)
#then we save the results to the table
_.save_results(con, probs, 'num matches=301')


