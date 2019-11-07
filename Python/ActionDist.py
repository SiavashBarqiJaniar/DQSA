from numpy.random import choice

list_of_candidates = [6, 2, 3]
probability_distribution = [.3, .2, .5]
number_of_items_to_pick = 1

draw = choice(list_of_candidates, number_of_items_to_pick,
              p=probability_distribution)
u = [x*3 for x in list_of_candidates]
print(u)