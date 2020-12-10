from sklearn.base import TransformerMixin
import youtokentome as yttm
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
from tempfile import NamedTemporaryFile
import gensim.downloader as api
from nltk.corpus import stopwords
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.base import ClassifierMixin, BaseEstimator


class TokenizerWrapper(TransformerMixin):
    def __init__(self, tokenizer, func='tokenize', lowercase=True, join=False):
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        
    def fit(self, X, y=None, **kwargs):
        return self
    
    def transform(self, X, y=None, **kwargs):
        if self.lowercase:
            for i, text in enumerate((X)):
                X[i] = text.lower()
        return [self.tokenizer.tokenize(x) for x in (X)]

class WordPostprocessor(TransformerMixin):
    def __init__(self, join=False):
        self.join = join
        self.word_apply = None
        
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, tokenized_texts, y=None, **kwargs):
        if self.join:
            return [' '.join(self.word_apply(w) for w in sent) for sent in tokenized_texts]
        else:
            return [[self.word_apply(w) for w in sent] for sent in (tokenized_texts)]

class Stemmer(WordPostprocessor):
    def __init__(self, join=False):
        super().__init__(join)
        self.stemmer = PorterStemmer()
        self.word_apply = self.stemmer.stem

class Lemmatizer(WordPostprocessor):
    def __init__(self, join=False):
        super().__init__(join)
        self.lemmatizer = WordNetLemmatizer()
        self.word_apply = self.lemmatizer.lemmatize
    
class BPETokenizer(TransformerMixin):
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.bpe = None
        
    def fit(self, X, y=None, **kwargs):
        
        with NamedTemporaryFile() as f, NamedTemporaryFile() as g:
            with open(f.name, 'w') as f_desc:
                for s in X:
                    f_desc.write(s + '\n')
            self.bpe = yttm.BPE.train(data=f.name, model=g.name, 
                                      vocab_size=self.vocab_size)
        return self
    
    def transform(self, X, y=None, **kwargs):
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return self.bpe.encode(X, output_type=yttm.OutputType.SUBWORD)

class Embedder(TransformerMixin):
    """
        makes text embedding (as averaged or weighted averaged vector of words)
    """
    def __init__(self, embedding):
        self.vectors = embedding.vectors
        self.vocab = {k: i for i, k in enumerate(embedding.index2word)}
        self.embedding = {w: embedding.get_vector(w) for w in embedding.vocab}
        self.emb_dim = self.embedding[next(iter(self.embedding.keys()))].shape[0]

    def fit(self, X, y=None, **kwargs):
        return self
    
    def transform(self, texts, weights=None):
        res = []
        for text in texts:
            idx = np.array([self.vocab[w] for w in text if w in self.vocab])
            if len(idx) == 0:
                res.append(np.zeros(self.vectors.shape[1]))
            else:
                res.append(self.vectors[idx].mean(0))
        return np.row_stack(res)


class OnlineSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, ksvm=False, kernel='linear', l2=1., l1=0.,):
        self.ksvm = ksvm
        self.kernel = kernel
        self.l2 = l2
        self.l1 = l1
        self.vw = None
        
    def fit(self, X, y):
        self.vw = VWClassifier(**self.get_params(), loss_function='hinge')
        self.vw.fit(X, y)
        return self
    
    def predict(self, X):
        preds = self.vw.predict(X)
        del self.vw
        self.vw = None
        return preds

    
class FeaturePipeline(TransformerMixin):
    @staticmethod
    def ID(x):
        return x

    def __init__(self, tokenizer='base', postprocessor='lemma', vectorizer='bow',
                 vectorizer_max_ngram=3, vectorizer_max_df=0.99, vectorizer_min_df=0.0001):
        self._init(tokenizer=tokenizer, postprocessor=postprocessor, vectorizer=vectorizer, 
                   vectorizer_max_ngram=vectorizer_max_ngram, vectorizer_max_df=vectorizer_max_df, 
                   vectorizer_min_df=vectorizer_min_df)
    
    def _init(self, tokenizer='base', postprocessor='lemma', vectorizer='bow', 
              vectorizer_max_ngram=3, vectorizer_max_df=0.99, vectorizer_min_df=0.0001    
             ):
        """
        Args:
            tokenizer (str): either 'base' or 'bpe'
            postprocessor (str): either 'lemma', 'stem' or 'none', igroned if tokenizer is bpe
            vectorizer (str): either 'bow', 'tfidf', 'hashing', 'glove_emb', 'w2v_emb'
                
        """
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.vectorizer = vectorizer
        self.pipeline = []
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
        self.vectorizer_max_ngram = vectorizer_max_ngram
        self.vectorizer_max_df = vectorizer_max_df
        self.vectorizer_min_df = vectorizer_min_df
        if tokenizer == 'base':
            self.pipeline.append(('base_tokenizer', TokenizerWrapper(TweetTokenizer())))
            if postprocessor == 'stem':
                self.pipeline.append(('stemmer', Stemmer(
                   # join=vectorizer == 'bow'
                )))
            elif postprocessor == 'lemma':
                self.pipeline.append(('lemmatizer', Lemmatizer(
                    #join=vectorizer == 'bow'
                )))
            elif postprocessor == 'none':
                pass
            else:
                raise ValueError()
        elif tokenizer == 'bpe':
            self.pipeline.append(('bpe', BPETokenizer())) 
        else:
            raise ValueError()
        kwargs = {
            'ngram_range': (1, self.vectorizer_max_ngram),
            'analyzer': 'word',
            'tokenizer': FeaturePipeline.ID
        }
        if self.vectorizer == 'bow':
            name = 'bow'
            cls = CountVectorizer
            kwargs['binary'] = True
            kwargs['max_df'] = self.vectorizer_max_df
            kwargs['min_df'] = self.vectorizer_min_df
        elif self.vectorizer == 'tfidf':
            name = 'tfidf'
            cls = TfidfVectorizer
            kwargs['max_df'] = self.vectorizer_max_df
            kwargs['min_df'] = self.vectorizer_min_df
        elif self.vectorizer == 'hashing':
            name = 'hash'
            cls = HashingVectorizer
            kwargs['n_features'] = 50000
        if self.vectorizer in ('bow', 'tfidf', 'hashing'):
            self.pipeline.append((name, cls(
                    stop_words=self.stopwords,
                    lowercase=False, **kwargs
                )))
        if self.vectorizer == 'glove_emb':
            self.pipeline.append((
                'glove_emb', Embedder(api.load('glove-twitter-50'))
            ))
        elif self.vectorizer == 'w2v_emb':
            self.pipeline.append((
                'w2v_emb', Embedder(api.load('word2vec-google-news-300'))
            ))
        self.pipeline = Pipeline(self.pipeline)
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.pipeline.fit_transform(X, y=y, **kwargs)
    
    def transform(self, X, **kwargs):
        return self.pipeline.transform(X, **kwargs)
    
    def set_params(self, **params):
        super().set_params(**params)
        self._init(**params)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer