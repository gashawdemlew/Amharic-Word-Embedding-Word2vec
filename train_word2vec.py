# import modules & set up logging
import logging
from gensim.models import Word2Vec,KeyedVectors,Phrases
import os
from hyperparameter import (WordEmbeddingConfig,TrainConfig)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dirname="C:/Users/has/Desktop/artdata/stemfinal"
LEMATIZER_DIR="lemmataization"
class corpus_sentences(object):# accept sentence stored one per line in list of files inside defined directory
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding='utf8'):
                yield line.split() 



def detect_bigrams():
    sentences = corpus_sentences(dirname) # a memory-friendly iterator
    bigram_transformer = Phrases(sentences)
    print(list(bigram_transformer))
    #bigram_model = Word2Vec(bigram_transformer[sentences], size=100,window=5, min_count=5, workers=5) 
    #return bigram_model               

#detect_bigrams()
def load_am_word_vectors():
    if WordEmbeddingConfig.sg==0:
            model_type='CBOW'
    else:
        model_type='Skip-gram'        
    if os.path.exists(TrainConfig.model_name):
        print('Loading Word2Vec Amharic Pretrained '+model_type+' model with '+str(WordEmbeddingConfig.emb_dim)+' dimension\n') 
        am_model= KeyedVectors.load(TrainConfig.model_name)
    else:
        print('Loading Sentences with memory freindly iterator ...\n')
        if WordEmbeddingConfig.USE_LEMA:
            sentences = corpus_sentences(LEMATIZER_DIR) # a memory-friendly iterator
        else:
            #sentences = corpus_sentences(DirConfig.PREPROCESSED_DIR) # a memory-friendly iterator
            sentences = corpus_sentences(dirname) # a memory-friendly iterator
        print('Training Word2Vec '+model_type+' with '+str(WordEmbeddingConfig.emb_dim)+' dimension\n') 
        am_model = Word2Vec(sentences, size=WordEmbeddingConfig.emb_dim, window=WordEmbeddingConfig.window, 
                            min_count=WordEmbeddingConfig.minFreq, workers=WordEmbeddingConfig.nthread,sg=WordEmbeddingConfig.sg,
                            iter=WordEmbeddingConfig.iterate,negative=WordEmbeddingConfig.negative,
                            hs=WordEmbeddingConfig.hs,sample=WordEmbeddingConfig.sample)
      
        #trim unneeded model memory = use (much) less RAM
        am_model.init_sims(replace=True)
        
        #Saving model    
        am_model.wv.save_word2vec_format(TrainConfig.model_name)        
        
    return am_model            

load_am_word_vectors()
#am_model=KeyedVectors.load_word2vec_format('C:/Users/has/Desktop/artdata/Artdata_sampling_Embedding.txt')
#print(am_model.most_similar('ተወነች',topn=5))
#print(am_model.most_similar('ቤተ',topn=5))
#print(am_model.most_similar('ዘገባ',topn=10))
#print(am_model.most_similar('የቋንቋ',topn=10))
#print(am_model.wv['ተወነች'])





























"""model = gensim.models.Word2Vec(cnt,size=300,alpha=0.025,window=5,min_count=10,workers=5,min_alpha=0.0001,sg=0,cbow_mean=1,sorted_vocab=1)
# summarize the loaded model
#print('model is ssss \n',model)
# summarize vocabulary
#class gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0, hashfxn=, iter=1, null_word=0, trim_rule=None, sorted_vocab=1)

#print('List of learned words are\n\n',words)
# access vector for one word
#print(model['sentence'])
# save model
model.wv.save_word2vec_format('w2v_bible.txt')
model.wv.save_word2vec_format('w2v_bible.model')
model.wv.save_word2vec_format('w2v_bible.vec')
model.wv.save_word2vec_format('w2v_bible.bin')
#model.save('w2v_python.txt')
# load model
my_test =KeyedVectors.load_word2vec_format('C:/Users/Pc/w2v_pythonnn.vec',encoding='utf-8',unicode_errors='ignore',limit=1000)  
words = list(my_test.wv.vocab)
X = my_test[my_test.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(my_test.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

#word2vec.Word2Vec(c, size=100, window=5, min_count=1, workers=4)"""