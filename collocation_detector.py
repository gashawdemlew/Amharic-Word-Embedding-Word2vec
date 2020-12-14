from nltk import BigramCollocationFinder
import nltk.collocations 
import io
import re
import os

#parameters
#corpus: is input data 
def tokenize(corpus):
     print('Tokenization ...')
     all_tokens=[]
     for fname in os.listdir(corpus):
        with open(os.path.join(corpus, fname),encoding='utf8') as fin:
           for sentence in fin: 
              tokens=re.compile('[\s+]+').split(sentence)
              all_tokens.extend(tokens)           
     return all_tokens
def collocation_finder(tokens,bigram_dir):

	bigram_measures = nltk.collocations.BigramAssocMeasures()
	
	#Search for bigrams with in a corpus
	finder = BigramCollocationFinder.from_words(tokens)
	
	#filter only Ngram appears morethan 3+ times
	finder.apply_freq_filter(10)
	
	frequent_bigrams = finder.nbest(bigram_measures.chi_sq,200) # chi square computer 
	#print(frequent_bigrams)
	PhraseWriter = io.open(bigram_dir, "w", encoding="utf8")
	
	for bigram in frequent_bigrams:
		PhraseWriter.write(bigram[0]+' '+bigram[1] + "\n")

def normalize_multi_words(tokenized_sentence,bigram_dir,corpus):
  
	bigram=set()
	sent_with_bigrams=[]
	index=0
	if not os.path.exists(bigram_dir):
		collocation_finder(tokenize(corpus),bigram_dir)
		#calling itsef
		normalize_multi_words(tokenized_sentence,bigram_dir,corpus)
	else:
		text=open(bigram_dir,encoding='utf8')
	   
		for line in iter(text):
		   line=line.strip()
		   if not line:  # line is blank
			   continue
		   else:
			   bigram.add(line)
		if len(tokenized_sentence)==1:
			sent_with_bigrams=tokenized_sentence
		else:
			while index <=len(tokenized_sentence)-2:
				mword=tokenized_sentence[index]+' '+tokenized_sentence[index+1]
				if mword in bigram:
					sent_with_bigrams.append(tokenized_sentence[index]+''+tokenized_sentence[index+1])
					index+=1
				else:
					sent_with_bigrams.append(tokenized_sentence[index])
				index+=1
				if index==len(tokenized_sentence)-1:
					sent_with_bigrams.append(tokenized_sentence[index])
			
		return sent_with_bigrams
		
#calling a method
#text_input: is list of sentence(not tokenized) : is same with corpus and you can re-organize it		
dirname="C:/Users/has/.spyder-py3/wikiextractor/text/wiki_012222"
dirname1="C:/Users/has/.spyder-py3/wikiextractor/text/AA"
text_input=open(dirname,encoding='utf8')
for sentence in text_input:
    tokens=re.compile('[\s+]+').split(sentence)
    normalized_token=[]
    MULTI_DIR="C:/Users/has/.spyder-py3/wikiextractor/text/bigram.txt"
    multi_words=normalize_multi_words(tokens,MULTI_DIR, dirname1)      
    #print("multi_words", multi_words)