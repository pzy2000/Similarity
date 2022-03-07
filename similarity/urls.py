from django.urls import path

from . import word2vec_similarity_catalog
from similarity.src.metadata import metadata_associate
from similarity.src.recommend import recommend_catalog
from . import bert_model
from . import recommend


urlpatterns = [
    path('init_model_vector/', recommend.init_model_vector, name='init_model_vector'),
    path('init_data_path/', metadata_associate.init_data_path, name='init_data_path'),
    path('add_metadata/', metadata_associate.add_metadata, name='add_metadata'),
    path('single_match/', metadata_associate.single_match, name='single_match'),
    path('recommend_catalog/', recommend_catalog.catalog_recommend, name='recommend_catalog'),
    path('multiple_match/', recommend.multiple_match, name='multiple_match'),
    path('get_state/', word2vec_similarity_catalog.get_state, name='get_state'),
    path('add_bert_corpus/', bert_model.add_corpus, name='add_bert_corpus'),
    path('get_bert_pretrain_state/', bert_model.get_pretrain_state, name='get_bert_pretrain_state'),
    path('get_bert_train_state/', bert_model.get_train_state, name='get_bert_train_state'),
    path('get_bert_retrain_state/', bert_model.get_retrain_state, name='get_bert_retrain_state'),
    path('get_bert_model_config/', bert_model.get_model_config, name='get_bert_model_config'),
    path('config_bert_model/', bert_model.config_model, name='config_bert_model'),
    path('add_bert_data/', bert_model.add_model_data, name='add_bert_data'),
    path('train_bert_model/', bert_model.train_model, name='train_model'),
    path('retrain_bert_model/', bert_model.train_re_model, name='retrain_model'),
    path('pretrain_bert_model/', bert_model.do_pretrain, name='pretrain_model'),
    path('increment_business_data/', recommend.increment_business_data, name='increment_business_data'),
    path('delete_business_data/', recommend.delete_business_data, name='delete_business_data')
]