from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

corpus_path = 'corpus.txt'
sentences = LineSentence(corpus_path)

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

model.save('word2vec_model.bin')