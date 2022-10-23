import random
import NeuralNetwork

max_gen = 100
n_prnts = 25
n_prnts_zero = 100
generation_number = 0
number_of_weights = 10
leftover_indexes = []

for i in range (number_of_weights):
    leftover_indexes.append(i)


#sort the list population by its rank
#return sorted list of tuples (model,score)
def rank_population(population_rank_list):
        return(sorted(population_rank_list, key = lambda x: x[1]))

while generation_number < max_gen:
    #cut the population to only have the best n
    survivor_population = rank_population[:n_prnts]
    #for each parent select a random new parent
    current_index = 0
    for parent in survivor_population:
        index_of_rand = random.randint(current_index, len(survivor_population)-1)
        number_of_inheritors = random.randint(1,number_of_weights-1)
        inherited_indexes = []
        #for i in range(number_of_inheritors):

class Evolution():
    def __int__(self):
        self.max_generations = 100
        self.initial_population_size = 100
        self.survivor_population_size = 50
        self.agent_list = []
        for amount in range(self.initial_population_size):
            self.agent_list[amount] = NeuralNetwork.NeuralNetwork()
        print(self.agent_list[0])