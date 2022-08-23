''' By Thierry N. and Giovanni. This code is used to generate consensus in BA model using qualities in the modify majority update rule.
We have considered One opinion and 500(niter) networks each formed by 100 nodes. '''

import networkx as nx
import matplotlib.pyplot as plt
import random
from random import sample
import numpy as np
import pylab
import csv
import pandas as pd
import statistics
import os


# This is the class to model my agent and its opinion
class MyAgent:
    # the opinion of the agent, either 0 or 1 (-1 means that the agent has not been initialised)
    opinion = -1
    # a boolean to say if an agent is a zealot or not
    zealot = False
    
    def __init__(self, opinion):
        self.opinion = opinion


n = 100
Tmax = 45000
niter = 50

random.seed(1234567890)


Datas = []

################ Lists
List_zealots = [0, 2, 5, 10]
Liste_m = [2, 4, 8, 16, 30]# [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 1]
Liste_al = [0.001,  0.5, 1,  1.5]#[0.001]#,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
Liste_qb = [0.7, 0.8, 0.9, 1]

Qa = 1 #Quality of opinions 0
proportion_of_agent_with_opinion_zero = 0.5

##################################### NETWORK CHOICE #####################
#### We use --BA-- for Barabasi-Albert Model, --ER-- for Erdos-Renyi Model and --WS-- for Watt-Strogatz Model
Network_name = ['BA']


########################### ZEALOTS #######################################
for Zeal in List_zealots:
    print(Zeal)
    ##########
    number_of_zealots_with_opinion_zero = Zeal #Number of Zealots with opinion 0
    number_of_zealots_with_opinion_one = 0 #Number of Zealots with opinion 1

    Threshold = float(n - number_of_zealots_with_opinion_one)/float(n)


    #############################################################################
    debug = False

    for Qb in Liste_qb:
        qualities=[Qa, Qb]
        #print(Qb)
        

        for al in Liste_al:

            for m in Liste_m:
                
                T = []
                
                j0 = 0
                j1 = 0
                jn = 0

                AA = Liste_m.index(m)


                for i in range(0,niter):
                    random.seed(i + AA*niter)
                    
                    ########## Build network
                    
                    for i1 in Network_name:
                        if i1 == 'BA':
                            connected_net = False
                            while not connected_net:
                                G = nx.barabasi_albert_graph(n,m)
                                connected_net = nx.is_connected(G)

                        if i1=='ER':
                            connected_net = False
                            while not connected_net:
                                G = nx.erdos_renyi_graph(n,p)
                                connected_net = nx.is_connected(G)

                        if i1=='WS':
                            connected_net = False
                            while not connected_net:
                                G = nx.watt_strogatz_graph(n,m,p)
                                connected_net = nx.is_connected(G)

                    ########## Computing centrality ############################
                    Degcentrality = nx.degree_centrality(G)
                    #Descending order sorting centrality
                    DegCent_sorted=dict(sorted(Degcentrality.items(), key=lambda item: item[1],reverse=True))
                    #Getting indices
                    Ind_cent=list(DegCent_sorted)[0:Zeal]

                        
                    # create the population (initialisation!!)
                    population = []
                    number_of_non_zealots = n - (number_of_zealots_with_opinion_zero+number_of_zealots_with_opinion_one)
                    for a in range(number_of_non_zealots):
                        the_opinion_of_the_guy = 0 if (a < (number_of_non_zealots*proportion_of_agent_with_opinion_zero)) else 1 
                        newguy = MyAgent(the_opinion_of_the_guy)
                        population.append(newguy)
                        
                    # This code is just for debug to count the num of guys and their opinions
                    if debug:
                        num_zeros=0
                        num_ones=0
                        for a1 in population:
                            if a1.opinion==0:
                                num_zeros+=1
                            else:
                                num_ones+=1
                        print(num_zeros)
                        print(num_ones)
                    
                    # Let's add the zealots to the population
                    for z0 in Ind_cent: #range(number_of_zealots_with_opinion_zero): 
                        newguy = MyAgent(0)
                        newguy.zealot = True
                        population.append(newguy)
                    for z1 in range(number_of_zealots_with_opinion_one): 
                        newguy = MyAgent(1)
                        newguy.zealot = True
                        population.append(newguy)
                        
                    np.random.shuffle(population)
                    
                    

                    ####################################################
                    cond = 0
                    t = 0
                    
                    while (cond == 0):
                        random_node = random.sample(list(G.nodes()), 1)[0]
                        
                        ### We need to verify if the selected agent is not a zealot
                        if population[random_node].zealot:
                            # do nothing!
                            pass
                        else:
                            # List of neighbors
                            neighbors = list(G.neighbors(random_node))
                            # Neighbor Opinions
                            vocal_neighbors = []
                            for neigh in neighbors:
                                if random.random() < qualities[ population[neigh].opinion ]:
                                    vocal_neighbors.append( population[neigh].opinion )

                            if len(vocal_neighbors)>0:
                                # Fraction of opinions one in the list
                                Nombre_ones_select = np.sum(vocal_neighbors) / float(len(vocal_neighbors))

                                #print(Nombre_ones_select)   
                                if Nombre_ones_select > 0.5:
                                    prob = (abs(Nombre_ones_select - 0.5) ** al) * (2 ** (al - 1)) + 0.5
                                        
                                else:
                                    prob = -(abs(Nombre_ones_select - 0.5) ** al) * (2 ** (al - 1)) + 0.5
                                    
                                Rand_num = random.random()
                                if Rand_num < prob:
                                    population[random_node].opinion = 1
                                else:
                                    population[random_node].opinion = 0
                                        

                        number_zeros=0
                        number_ones=0
                        for b in population:
                            if b.opinion==0:
                                number_zeros+=1
                            else:
                                number_ones+=1
                        
                        #################################################################
                        if t < Tmax:
                            if (float(number_zeros)/float(n) >= Threshold):
                                T.append(t)
                                j0 = j0 + 1
                                cond = 1

                            elif (float(number_ones)/float(n) >= float(n - Zeal)/float(n) ):
                                j1 = j1 + 1
                                T.append(t)
                                cond = 1

                            else:
                                t = t + 1
                        else:
                            cond = 1
                            jn = jn + 1

                Tm = np.mean(T)
                Tsd = np.std(T)
                Tmin = Tm - Tsd
                Tmaxi = Tm + Tsd

               

    ######################################################################
    ##################### Save data ######################################
    ######################################################################

                filename = "Results/Time"+"_Zeal"+str(Zeal)+"_qb"+str(Qb)+"_al"+str(al)+"_m"+str(m)+".csv"
                os.makedirs("Results",exist_ok = True)
                file=open(filename,'w')
                write = csv.writer(file,delimiter ='\n') 
                write.writerow(T)
                file.close()


                

                datas = [number_of_zealots_with_opinion_zero, number_of_zealots_with_opinion_one, Qa, Qb, al, m, j0, j1, jn, Tm, Tsd, Tmin, Tmaxi]
                new_lst = str(datas)[1:-1] 
                Datas.append(new_lst)
                filename_1 = "Results/data.csv"
                file_1=open(filename_1,'w')
                write = csv.writer(file_1, delimiter ='\n') 
                write.writerow(Datas)
                file_1.close()

               
######################################################################
##################### End Save data ##################################
######################################################################

print('Thanks, the simulation is finished')
