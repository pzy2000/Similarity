from django.urls import path
from . import recommend

urlpatterns = [
    path('init_model_vector/', recommend.init_model_vector, name='init_model_vector'),
    path('multiple_match/', recommend.multiple_match, name='multiple_match'),
    path('increment_business_data/', recommend.increment_business_data, name='increment_business_data'),
    path('delete_business_data/', recommend.delete_business_data, name='delete_business_data'),

]
