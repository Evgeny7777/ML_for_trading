from skopt.space import Integer, Real
from py.my_pd_utils import make_list_if_single

def make_search_space(space_dictionary):
    # Read search space
    search_space_dict = {k:v for k,v in space_dictionary.items() if len(make_list_if_single(v)) > 1}
    search_space = [Integer(val[0], val[1], name=key) 
                for key,val in search_space_dict.items() 
                if isinstance(val[0], int)]

    search_space = search_space + [Real(val[0], val[1], name=key) 
                for key,val in search_space_dict.items() 
                if isinstance(val[0], float)]
    return search_space