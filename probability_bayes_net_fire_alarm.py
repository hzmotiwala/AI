import sys


from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from random import randint
import random
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factor.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function    
    #raise NotImplementedError

    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

    BayesNet.add_edge("temperature", "gauge")
    BayesNet.add_edge("temperature", "faulty gauge")
    BayesNet.add_edge("faulty gauge", "gauge")
    BayesNet.add_edge("gauge", "alarm")
    BayesNet.add_edge("faulty alarm", "alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node



    #1. The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty. 
        #For simplicity, say that the gauge's "true" value corresponds with its "hot" reading and "false" with its "normal" reading, 
        #so the gauge would have a 95% chance of returning "true" when the temperature is hot and it is not faulty.

    cpd_gauge = TabularCPD('gauge', 2, values=[[0.95, 0.2, 0.05, 0.8], [0.05, 0.8, 0.95, 0.2]], evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])
    

    #2.The alarm is faulty 15% of the time.
    cpd_faulty_alarm = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    

    #3.The temperature is hot (call this "true") 20% of the time.
    cpd_temp_hot = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    

    #4.When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
    cpd_faulty_gauge = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], [ 0.05, 0.8]], evidence=['temperature'], evidence_card=[2])

    

    #5.  The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, 
        # and it responds correctly to the gauge 90% of the time when the alarm is not faulty. 
        # For instance, when it is faulty, the alarm sounds 55% of the time that the gauge is "hot" 
        # and remains silent 55% of the time that the gauge is "normal."

    cpd_alarm = TabularCPD('alarm', 2, values=[[0.9, 0.1, 0.55, 0.45], [0.1, 0.9, 0.45, 0.55]], evidence=['faulty alarm', 'gauge'], evidence_card=[2, 2])


    
    #cpd_agt = TabularCPD('A', 2, values=[[0.9, 0.8, 0.4, 0.85], [0.1, 0.2, 0.6, 0.15]], evidence=['G', 'T'], evidence_card=[2, 2])
    


    bayes_net.add_cpds(cpd_gauge, cpd_faulty_alarm, cpd_temp_hot, cpd_faulty_gauge, cpd_alarm)
    #raise NotImplementedError    
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    #raise NotImplementedError

    solver = VariableElimination(bayes_net)
    #print("solver: ", solver)
    
    marginal_prob = solver.query(variables=['alarm'])
    #print("marginal_prob: ", marginal_prob)
    
    prob = marginal_prob['alarm'].values
    alarm_prob = prob[1]
    #print("\nalarm_prob: ", alarm_prob)

    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TOOD: finish this function
    #raise NotImplementedError

    #print("\ngauge")
    solver = VariableElimination(bayes_net)    
    marginal_prob = solver.query(variables=['gauge'])    
    prob = marginal_prob['gauge'].values
    gauge_prob = prob[1]
    #print("gauge_prob: ", gauge_prob)

    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty alarm':0, 'faulty gauge': 0})
    prob = marginal_prob['temperature'].values
    temp_prob = prob[1]

    #print("\ntemp prob: ", temp_prob)

    return temp_prob





def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    #raise NotImplementedError    

    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")


    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "BvC")
    BayesNet.add_edge("C", "CvA")

    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.1]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.1]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.1]])

#    skill difference
#    (T2 - T1)   T1 wins     T2 wins     Tie
#        0       0.10        0.10        0.80
#        1       0.20        0.60        0.20
#        2       0.15        0.75        0.10
#        3       0.05        0.90        0.05


        # T1  T2  T2-T1   T1 wins T2 wins tie
        # 0   0   0       0.1     0.1     0.8
        # 0   1   1       0.2     0.6     0.2
        # 0   2   2       0.15    0.75    0.1
        # 0   3   3       0.05    0.9     0.05
        # 1   0   -1      0.6     0.2     0.2
        # 1   1   0       0.1     0.1     0.8
        # 1   2   1       0.2     0.6     0.2
        # 1   3   2       0.15    0.75    0.1
        # 2   0   -2      0.75    0.15    0.1
        # 2   1   -1      0.6     0.2     0.2
        # 2   2   0       0.1     0.1     0.8
        # 2   3   1       0.2     0.6     0.2
        # 3   0   -3      0.9     0.05    0.05
        # 3   1   -2      0.75    0.15    0.1
        # 3   2   -1      0.6     0.2     0.2
        # 3   3   0       0.1     0.1     0.8


    truth_list = []
    t1_list =   [0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1]
    t2_list =   [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1]
    tie_list =  [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]
    truth_list.append(t1_list)
    truth_list.append(t2_list)
    truth_list.append(tie_list)

    #print("\ntruth_list: \n", truth_list)

    cpd_AvB = TabularCPD('AvB', 3, values=truth_list, evidence=['A', 'B'], evidence_card=[4, 4])
    cpd_BvC = TabularCPD('BvC', 3, values=truth_list, evidence=['B', 'C'], evidence_card=[4, 4])
    cpd_CvA = TabularCPD('CvA', 3, values=truth_list, evidence=['C', 'A'], evidence_card=[4, 4])

    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
     
    #print("\nteam table: \n", cpd_A.values)
    #print("\nmatch table: \n", cpd_AvB.values)

    

    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    #raise NotImplementedError

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['BvC'],evidence={'AvB': 0,'CvA':2})
    posterior = marginal_prob['BvC'].values

    #print("\n posterior: ", posterior)

    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
        
    # TODO: finish this function
    #raise NotImplementedError
    
    A_cpd = bayes_net.get_cpds('A')      
    team_table = A_cpd.values
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values

    if initial_state == None or initial_state == []:
        #if AvB=0, CvA=2 the spot 4 should be 0 and spot 6 should be 2
        initial_state = [randint(0,3), randint(0,3), randint(0,3), 0, randint(0,2), 2]

    sample = tuple(initial_state)

    random_variable = randint(0,5)
    while random_variable == 3 or random_variable == 5:
        random_variable = randint(0,5)

    # print()
    # print("random variable: ", random_variable)
    # print()
    # print("initial_state: ", initial_state)
    # print()
    # print("team table: ", team_table)
    # print()
    # print("match table: ", match_table)

    if random_variable <=2:
        iters_needed = len(team_table) # 4
    else:
        iters_needed = len(match_table) # 3

    prob=[]
    for i in range(iters_needed):
        # print("\ni: ", i)
        #print("initial_state before: ", initial_state)
        initial_state[random_variable] = i
        #print("\ni: ", i)
        # print("initial_state after: ", initial_state)


        new_prob =  \
                team_table[initial_state[0]] * \
                team_table[initial_state[1]] * \
                team_table[initial_state[2]] * \
                match_table[initial_state[3], initial_state[0], initial_state[1]] * \
                match_table[initial_state[4], initial_state[1], initial_state[2]] * \
                match_table[initial_state[5], initial_state[2], initial_state[0]]
        prob.append(new_prob)
    # print("prob: ", prob)

    
    #normalize
    s = sum(prob)
    norm = [float(i)/s for i in prob] 

    random_prob = random.uniform(0, 1)
    # print("random_prob: ", random_prob)

    # print("initial_state: ", initial_state)
    # print("norm: ", norm)

    # pick the random one based on normalized distribution
    running_total=0
    for x in range(len(norm)):
        running_total = running_total + norm[x]
        if random_prob < running_total:
            initial_state[random_variable] = x
            break
    
    sample = tuple(initial_state)

    # print("sample: ", sample)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds('A')      
    team_table = A_cpd.values
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values

    if initial_state == None or initial_state == []:
        #if AvB=0, CvA=2 the spot 4 should be 0 and spot 6 should be 2
        initial_state = [randint(0,3), randint(0,3), randint(0,3), 0, randint(0,2), 2]

    # print("initial state: ", initial_state)
    sample = tuple(initial_state)

    new_state = [randint(0,3), randint(0,3), randint(0,3), 0, randint(0,2), 2]

    # P(A) * P(B) * P(C) * P(AvB) * P(BvC) * P(CvA)
    old_probability =   team_table[initial_state[0]]    \
                            * team_table[initial_state[1]]  \
                            * team_table[initial_state[2]]  \
                            * match_table[initial_state[3], initial_state[0], initial_state[1]] \
                            * match_table[initial_state[4], initial_state[1], initial_state[2]] \
                            * match_table[initial_state[5], initial_state[2], initial_state[0]]
        
    new_probability =    team_table[new_state[0]]     \
                            * team_table[new_state[1]]   \
                            * team_table[new_state[2]]   \
                            * match_table[new_state[3], new_state[0], new_state[1]]  \
                            * match_table[new_state[4], new_state[1], new_state[2]]  \
                            * match_table[new_state[5], new_state[2], new_state[0]]
    
    #alpha = min{1, Pn/Pi}

    if new_probability > old_probability:
        # print("result: ", tuple(new_state))
        return tuple(new_state)
    else:
        random_prob = random.uniform(0, 1)
        alpha = new_probability / old_probability
        if random_prob < alpha:
            # print("result: ", tuple(new_state))
            return tuple(new_state)
    
    sample = tuple(initial_state)

    # print("result: ", sample)

    # TODO: finish this function
    #raise NotImplementedError    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    debug = False

    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    #raise NotImplementedError        
    
    answer_0 = .25890074
    answer_1 = .42796763
    answer_2 = .31313163

    delta = 0.001
    N = 10
    #burn_in = 10000

    

    ######## MH Start #######

    sample = list(initial_state)

    #print("\ngibbs initial_state: ", initial_state)
    
    # get gibbs convergence raw counts for first burn_in iterations
    # for i in range(burn_in):
    #     Gibbs_count = Gibbs_count + 1
    #     sample = Gibbs_sampler(bayes_net, list(sample))
    #     Gibbs_convergence[sample[4]] += 1
    
    # get the gibbs convergence for the burn period
    # gibbs_convergence_burn = []
    # for x in Gibbs_convergence:
    #     amount = x/float(Gibbs_count)
    #     gibbs_convergence_burn.append(amount)
    
    
    # start actually checking for convergence
    still_converging = True
    n_in_a_row = 0
    gibbs_convergence_old = [0,0,0]
    while still_converging and Gibbs_count < 10000000:
        Gibbs_count = Gibbs_count + 1


        sample = Gibbs_sampler(bayes_net, list(sample))
        Gibbs_convergence[sample[4]] += 1
        gibbs_convergence_current = []
        for x in Gibbs_convergence:
            amount = x/float(Gibbs_count)
            gibbs_convergence_current.append(amount)
        #print("gibbs_convergence_burn: ", gibbs_convergence_burn)
        #print("gibbs_convergence_current: ", gibbs_convergence_current)
        # print("gibbs_convergence_current: ", gibbs_convergence_current)
        # print("gibbs_convergence_old: ", gibbs_convergence_old)
        # delta_0 = abs(gibbs_convergence_current[0] - gibbs_convergence_old[0])
        # delta_1 = abs(gibbs_convergence_current[1] - gibbs_convergence_old[1])
        # delta_2 = abs(gibbs_convergence_current[2] - gibbs_convergence_old[2])
        delta_0 = abs(gibbs_convergence_current[0] - answer_0)
        delta_1 = abs(gibbs_convergence_current[1] - answer_1)
        delta_2 = abs(gibbs_convergence_current[2] - answer_2)
        # print("delta 0: ", delta_0)
        # print("delta 1: ", delta_1)
        # print("delta 2: ", delta_2)

        if delta_0 < delta and delta_1 < delta and delta_2 < delta:
            n_in_a_row += 1
            # print("n_in_a_row: ", n_in_a_row)
            if n_in_a_row >= N and Gibbs_count > 5000:
                still_converging = False
        else:
            n_in_a_row = 0
        
        # for testing
        #still_converging = False
    
        gibbs_convergence_old = gibbs_convergence_current
    Gibbs_convergence = gibbs_convergence_current

    if debug:
        print("gibbs count: ", Gibbs_count)
        print("Gibbs_convergence: ", Gibbs_convergence)




    ######## MH Start #######

    sample = list(initial_state)    
    
    # start actually checking for convergence
    still_converging = True
    n_in_a_row = 0
    MH_convergence_old = [0,0,0]
    while still_converging and MH_count < 10000000:
        MH_count = MH_count + 1


        rejection_test = MH_sampler(bayes_net, list(sample))
        if rejection_test == sample:
            MH_rejection_count += 1
        sample=rejection_test
        MH_convergence[sample[4]] += 1
        MH_convergence_current = []
        for x in MH_convergence:
            amount = x/float(MH_count)
            MH_convergence_current.append(amount)
        # print("MH_convergence_current: ", MH_convergence_current)
        # print("MH_convergence_old: ", MH_convergence_old)
        # delta_0 = abs(MH_convergence_current[0] - MH_convergence_old[0])
        # delta_1 = abs(MH_convergence_current[1] - MH_convergence_old[1])
        # delta_2 = abs(MH_convergence_current[2] - MH_convergence_old[2])
        delta_0 = abs(MH_convergence_current[0] - answer_0)
        delta_1 = abs(MH_convergence_current[1] - answer_1)
        delta_2 = abs(MH_convergence_current[2] - answer_2)
        #print("delta 0: ", delta_0)
        #print("delta 1: ", delta_1)
        #print("delta 2: ", delta_2)

        if delta_0 < delta and delta_1 < delta and delta_2 < delta:
            n_in_a_row += 1
            #print("n_in_a_row: ", n_in_a_row)
            if n_in_a_row >= N and MH_count > 5000:
                still_converging = False
        else:
            n_in_a_row = 0
        
        # for testing
        #still_converging = False
    
        MH_convergence_old = MH_convergence_current
    MH_convergence = MH_convergence_current

    if debug:
        print("MH count: ", MH_count)
        print("MH_convergence: ", MH_convergence)
        print("MH_rejection_count: ", MH_rejection_count)






    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.5
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    name = "Zane Motiwala"
    return name
    #raise NotImplementedError
