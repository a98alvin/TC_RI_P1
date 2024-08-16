def multiple_conditions_index_finder(list_of_indices):
    from functools import reduce
    list_of_lists = list_of_indices # Put indices here to add to criteria
    return list(reduce(lambda a, b: set(a) & set(b), list_of_lists))