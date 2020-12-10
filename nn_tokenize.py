from torchtext.data import BucketIterator, Field, Example, Dataset, TabularDataset
import torch
import wrappers
from scipy.sparse import vstack
from transformers import BertTokenizer
from tqdm import tqdm
import random
import pandas as pd



class BaseDataset(object):
    def __init__(self, **kwargs):
        self.dataset = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.batch_size = kwargs.get('batch_size')
        self.device = kwargs.get('device')


    def make_splits(self):
        train, valid, test = self.dataset.split(split_ratio=[0.6, 0.2, 0.2], 
                                        stratified=True, strata_field='label', 
                                        random_state=random.getstate())
        return train, valid, test

    
    def make_iterators(self):
        train_iter = BucketIterator(self.train_data, batch_size=self.batch_size, 
                                    device=self.device, shuffle=True, 
                                    sort_key=lambda x: len(x.comment))
        valid_iter, test_iter = BucketIterator.splits((self.valid_data, self.test_data), 
                                                batch_size=self.batch_size, 
                                                device=self.device, shuffle=False, 
                                                sort_key=lambda x: len(x.comment))
        return {'train': train_iter, 
                'valid': valid_iter, 
                'test': test_iter}
    
    

class DatasetsFromVectorized(BaseDataset):
    def __init__(self, path, comment_col='comment', label_col='label', 
                 batch_size=32, device=torch.device('cpu'), make_iters=False):
        super().__init__(batch_size=batch_size, device=device)
        df = pd.read_csv(path, usecols=[label_col, comment_col])
        df.dropna(axis='index', how='any', inplace=True)

        pipeline = wrappers.FeaturePipeline(tokenizer='bpe', postprocessor='none', vectorizer='tfidf',  
                                            vectorizer_max_df=0.99, vectorizer_min_df=0.001)
        X = pipeline.fit_transform(df[comment_col].values)
        y = df[label_col].values

        self.text_field = Field(sequential=False, use_vocab=False, 
                                batch_first=True, postprocessing=lambda x, y: vstack(x).toarray(), dtype=torch.float)
        self.label_field = Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)
        self.fields = [('comment', self.text_field), ('label', self.label_field)]
        self.examples = self.make_examples(X, y)
        self.dataset = Dataset(self.examples, self.fields)

        self.train_data, self.valid_data, self.test_data = self.make_splits()
        if make_iters:
            self.iters = self.make_iterators()


    def make_examples(self, X, y):
        examples = []
        for x, y in zip(X, y):
            example = Example.fromlist([x, y], fields=self.fields)
            examples.append(example)
        return examples
    

class DatasetsTokenizeSimple(BaseDataset):
    def __init__(self, tokenize_func, path, format, comment_col='comment', 
                 label_col='label', lower=True, use_vocab=True, batch_size=128, device=torch.device('cpu'), 
                 make_iters=False):
        super().__init__(batch_size=batch_size, device=device)
        if not use_vocab:
            self.text_field = Field(use_vocab=use_vocab, batch_first=True,
                                tokenize=lambda x: tokenize_func(x), pad_token=1, unk_token=0)
        else:
            self.text_field = Field(use_vocab=use_vocab, batch_first=True,
                                tokenize=lambda x: tokenize_func(x), lower=lower)
            
        self.label_field = Field(sequential=False, use_vocab=False, batch_first=True,
                           preprocessing=lambda x: int(x), is_target=True)
        self.fields = {'label': (label_col, self.label_field), 
                      'comment': (comment_col, self.text_field)}

        self.dataset = TabularDataset(path=path, format=format, fields=self.fields)
        if use_vocab:
            self.text_field.build_vocab(self.dataset)

        self.batch_size = batch_size
        self.device = device
        self.train_data, self.valid_data, self.test_data = self.make_splits()
        if make_iters:
            self.iters = self.make_iterators()
    
    
    
class DatasetsTokenizeBert(BaseDataset):
    def __init__(self, path, pretrained_model_name, 
                 text_a='comment', text_b='parent_comment', label='label', 
                 batch_size=32, device=torch.device('cpu'), make_iters=False):
        super().__init__(batch_size=batch_size, device=device)
        df = pd.read_csv(path, usecols=[label, text_a, text_b])
        df.dropna(axis='index', how='any', inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        pad = self.tokenizer.pad_token_id
        pad1 = self.tokenizer.pad_token_type_id
        unk = self.tokenizer.unk_token

        self.input_ids = Field(sequential=True, use_vocab=False, batch_first=True, 
                               pad_token=pad, unk_token=unk,
                               preprocessing=lambda x: self.get_field(x[0], x[1], 'input_ids'))
        self.token_type_ids = Field(sequential=True, use_vocab=False, batch_first=True, 
                                    pad_token=pad1, unk_token=unk, 
                                    preprocessing=lambda x: self.get_field(x[0], x[1],'token_type_ids'))
        self.attention_mask = Field(sequential=True, use_vocab=False, batch_first=True, 
                                    pad_token=pad, unk_token=unk, 
                                    preprocessing=lambda x: self.get_field(x[0], x[1], 'attention_mask'))
        self.label = Field(sequential=False, use_vocab=False, batch_first=True,
                           preprocessing=lambda x: int(x), is_target=True)

        self.fields = {'label': ('label', self.label), 
                       'text': [('comment', self.input_ids), 
                                ('token_type_ids', self.token_type_ids), 
                                ('attention_mask', self.attention_mask)]}

        self.examples = self.make_examples(df.values)
        self.dataset = Dataset(self.examples, fields=[self.fields['label']] + self.fields['text'])

        self.train_data, self.valid_data, self.test_data = self.make_splits()
        if make_iters:
            self.iters = self.make_iterators()


    
    def get_field(self, x, y, field):
        return self.tokenizer.encode_plus(text=x, text_pair=y, 
                                   return_token_type_ids=True, 
                                   return_attention_mask=True, 
                                   truncation=True, max_length=256)[field]


    def make_examples(self, data):
        examples = []
        with tqdm(total=len(data)) as pbar:
            for row in data:
                example_dict = {'label': row[0], 'text': [row[1].lower(), row[2].lower()]}
                example = Example.fromdict(example_dict, fields=self.fields)
                examples.append(example)
                pbar.update(1)
        return examples      
