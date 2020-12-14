class WordEmbeddingConfig(object):
    """Word2Vec Training parameters"""
    window=10 #Maximum skip length window between words
    emb_dim=300 # Set size of word vectors
    emb_lr=0.05 #learning rate for SGD estimation.
    nepoach=30 #number of training epochs
    nthread=12 #number of training threads
    sample = 1e-5 #Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled
    negative = 10 #negative sampling is used with defined negative example
    hs = 0 #0 Use Hierarchical Softmax; default is 0 (not used)
    binary=0 # 0 means not saved as .bin. Change to 1 if allowed to binary format
    sg=1 # 0 means CBOW model is used. Change to 1 to use Skip-gram model
    iterate=5 # Run more training iterations
    minFreq=3 #This will discard words that appear less than minFreq times  
    USE_LEMA=0 #not using lemmataization 
    EMBEDDING_DIR="C:/Users/has/Desktop/artdata/Artdata_sampling_NoStop_1e-5_Embedding" 
class TrainConfig(object):
    sg=WordEmbeddingConfig.sg
    DIM=WordEmbeddingConfig.emb_dim
    if sg==0:
        model_name='am_word2vec_cbow_'+str(DIM)+'D'
    elif sg==1:
         model_name='am_word2vec_sg_'+str(DIM)+'D'
    if WordEmbeddingConfig.USE_LEMA:
        model_name='lema_'+model_name
        
    if WordEmbeddingConfig.binary==0:
        model_name=WordEmbeddingConfig.EMBEDDING_DIR+'.txt'
    else:
        model_name=WordEmbeddingConfig.EMBEDDING_DIR+'.bin'
        
