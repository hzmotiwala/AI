import numpy as np


def part_1_a() -> tuple:
    """Provide probabilities for the letter HMMs outlined below.

    Letters Y and Z.

    See README.md for example probabilities for the letter A.
    See README.md for expected HMMs probabilities.
    See README.md for tuple of states.

    Returns:
        ( prior probabilities for all states for letter Y,
          transition probabilities between states for letter Y,
          emission probabilities for all states for letter Y,
          prior probabilities for all states for letter Z,
          transition probabilities between states for letter Z,
          emission probabilities for all states for letter Z )

        Sample Format (not complete):

        ( {'Y1': prob_of_starting_in_Y1, ...},
          {'Y1': {'Y1': prob_of_transition_from_Y1_to_Y1,
                  'Y2': prob_of_transition_from_Y1_to_Y2}, ...},
          {'Y1': [prob_of_observing_0, prob_of_observing_1], ...},
          {'Z1': prob_of_starting_in_Z1, ...},
          {'Z1': {'Z1': prob_of_transition_from_Z1_to_Z1,
                  'Z2': prob_of_transition_from_Z1_to_Z2}, ...},
          {'Z1': [prob_of_observing_0, prob_of_observing_1], ...} )
    """


    #raise NotImplementedError
    
    Y_states = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Yend']
    Z_states = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Zend']

    """Letter Y"""
    # prior probabilities for all states for letter Y
    y_prior_probs = {'Y1' : 1.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0}

    # transition probabilities between states for letter Y
    y_transition_probs = {'Y1' : {'Y1' : 0.667,
                             'Y2' : 0.333,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y2' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 1.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y3' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 1.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y4' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 1.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y5' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.667,
                             'Y6' : 0.333,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y6' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 1.0,
                             'Yend' : 0.0},
                    'Y7' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.667,
                             'Yend' : 0.333},
                    'Yend' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 1.0}}

    # emission probabilities for all states for letter Y
    y_emission_probs = { 'Y1' : [0.0, 1.0],
                         'Y2' : [1.0, 0.0],
                         'Y3' : [0.0, 1.0],
                         'Y4' : [1.0, 0.0],
                         'Y5' : [0.0, 1.0],
                         'Y6' : [1.0, 0.0],
                         'Y7' : [0.0, 1.0],
                         'Yend' : [0.0, 0.0]}

    """Letter Z"""
    # prior probabilities for all states for letter Z
    z_prior_probs = {'Z1' : 1.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0}

    # transition probabilities between states for letter Z
    z_transition_probs = {'Z1' : {'Z1' : 0.667,
                             'Z2' : 0.333,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z2' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 1.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z3' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.667,
                             'Z4' : 0.333,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z4' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 1.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z5' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 1.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z6' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 1.0,
                             'Zend' : 0.0},
                    'Z7' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 1.0},
                    'Zend' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 1.0}}

    # emission probabilities for all states for letter Z
    z_emission_probs = { 'Z1' : [0.0, 1.0],
                         'Z2' : [1.0, 0.0],
                         'Z3' : [0.0, 1.0],
                         'Z4' : [1.0, 0.0],
                         'Z5' : [0.0, 1.0],
                         'Z6' : [1.0, 0.0],
                         'Z7' : [0.0, 1.0],
                         'Zend' : [0.0, 0.0]}

    return (y_prior_probs, y_transition_probs, y_emission_probs,
            z_prior_probs, z_transition_probs, z_emission_probs)


def viterbi(evidence_vector: list, states: list, prior_probs: dict, 
        transition_probs: dict, emission_probs: dict) -> tuple:
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list(int)): List of 0s (Silence) or 1s (Dot/Dash).
            example: [1, 0, 1, 1, 1]
        states (list(string)): List of all states.
            example: ['A1', 'A2', 'A3', 'Aend']
        prior_probs (dict): prior distribution for each state.
            example: {'A1'  : 1.0,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}
        transition_probs (dict): dictionary representing transitions from
            each state to every other state, including self.
            example: {'A1'  : {'A1'  : 0.0,
                               'A2'  : 1.0,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 1.0,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}
        emission_probs (dict): dictionary of probabilities of outputs from
            each state.
            example: {'A1'  : [0.0, 1.0],
                      'A2'  : [1.0, 0.0],
                      'A3'  : [0.0, 1.0],
                      'Aend': [0.0, 0.0]}

    Returns:
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

        Example:
            ( ['A1', 'A2', 'A3', 'A3', 'A3'],
              0.445 )
    """

    #raise NotImplementedError

    #print("evidence_vector: ", evidence_vector)
    #print("")
    debug = False

    if debug:
        print("states: ", states)
        print("evidence_vector: ", evidence_vector)

    states_len = len(states)
    input_len=len(evidence_vector)
    Trelis = np.zeros((states_len,input_len))
    #Trelis_pointer = np.chararray((states_len,input_len), itemsize=2)
    Trelis_pointer = np.zeros((states_len,input_len), dtype=int)

    for i in range(states_len):
        Trelis[i][0]=prior_probs[states[i]] * emission_probs[states[i]][evidence_vector[0]]
        Trelis_pointer[i][0] = 0

    for transition in range(1, input_len):
        for i in range(states_len):
            best_transition_prob = Trelis[0][transition-1]*transition_probs[states[0]][states[i]]
            pointer = 0
            for z in range(1,states_len):
                other_transition_probs = Trelis[z][transition-1]*transition_probs[states[z]][states[i]]
                if other_transition_probs > best_transition_prob:
                    best_transition_prob = other_transition_probs
                    pointer = z

            best_prob = best_transition_prob * emission_probs[states[i]][evidence_vector[transition]]
            Trelis[i][transition] = best_prob
            Trelis_pointer[i][transition] = pointer

    if debug:
        print("\nTrelis end: \n", Trelis)
        print("\nTrelis pointer: \n", Trelis_pointer)
        print("pointer 2, 4: ", Trelis_pointer[2][4])


    # The highest probability
    sequence=[]
    probability = np.amax(Trelis, axis=0)[-1]
    backtrack = np.argmax(Trelis, axis=0)[-1]
    backtrack_state = states[backtrack]
    sequence.append(backtrack_state)

    for t in range(input_len - 1, 0, -1):
        backtrack = Trelis_pointer[backtrack][t]
        backtrack_state=states[backtrack]
        sequence.insert(0,backtrack_state)
    if debug:
        print()
        print("______________ new ____________")
        print("sequence: ", sequence)
        print("probability: ", probability)
        print("____________ end ______________")
        print()


    return sequence, probability


def part_2_a() -> tuple:
    """Provide probabilities for the NOISY letter HMMs outlined below.

    Letters A, Y, Z, letter pause, word space

    See README.md for example probabilities for the letter A.
    See README.md for expected HMMs probabilities.

    Returns:
        ( list of all states for letter A,
          prior probabilities for all states for letter A,
          transition probabilities between states for letter A,
          emission probabilities for all states for letter A,
          list of all states for letter Y,
          prior probabilities for all states for letter Y,
          transition probabilities between states for letter Y,
          emission probabilities for all states for letter Y,
          list of all states for letter Z,
          prior probabilities for all states for letter Z,
          transition probabilities between states for letter Z,
          emission probabilities for all states for letter Z,
          list of all states for letter pause,
          prior probabilities for all states for letter pause,
          transition probabilities between states for letter pause,
          emission probabilities for all states for letter pause,
          list of all states for word space,
          prior probabilities for all states for word space,
          transition probabilities between states for word space,
          emission probabilities for all states for word space )

        Sample Format (not complete):

        ( ['A1', ...],
          ['A1': prob_of_starting_in_A1, ...],
          {'A1': {'A1': prob_of_transition_from_A1_to_A1,
                  'A2': prob_of_transition_from_A1_to_A2}, ...},
          {'A1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['Y1', ...],
          ['Y1': prob_of_starting_in_Y1, ...],
          {'Y1': {'Y1': prob_of_transition_from_Y1_to_Y1,
                  'Y2': prob_of_transition_from_Y1_to_Y2}, ...},
          {'Y1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['Z1', ...],
          ['Z1': prob_of_starting_in_Z1, ...],
          {'Z1': {'Z1': prob_of_transition_from_Z1_to_Z1,
                  'Z2': prob_of_transition_from_Z1_to_Z2}, ...},
          {'Z1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['L1', ...]
          ['L1': prob_of_starting_in_L1, ...],
          {'L1': {'L1': prob_of_transition_from_L1_to_L1,
                  'L2': prob_of_transition_from_L1_to_L2}, ...},
          {'L1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['W1', ...]
          ['W1': prob_of_starting_in_W1, ...],
          {'W1': {'W1': prob_of_transition_from_W1_to_W1,
                  'W2': prob_of_transition_from_W1_to_W2}, ...},
          {'W1': [prob_of_observing_0, prob_of_observing_1], ...} )
        """


    #raise NotImplementedError

    """Letter A"""
    # expected states names for letter A
    a_states = ['A1', 'A2', 'A3', 'Aend']

    # prior probabilities for all states for letter A
    a_prior_probs = { 'A1'  : 0.333,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}

    # transition probabilities between states for letter A
    a_transition_probs = {'A1'  : {'A1'  : 0.2,
                                   'A2'  : 0.8,
                                   'A3'  : 0.0,
                                   'Aend': 0.0},
                          'A2'  : {'A1'  : 0.0,
                                   'A2'  : 0.2,
                                   'A3'  : 0.8,
                                   'Aend': 0.0},
                          'A3'  : {'A1'  : 0.0,
                                   'A2'  : 0.0,
                                   'A3'  : 0.667,
                                   'Aend': 0.111,
                                   'L1'  : 0.111,
                                   'W1'  : 0.111},
                          'Aend': {'A1'  : 0.0,
                                   'A2'  : 0.0,
                                   'A3'  : 0.0,
                                   'Aend': 1.0}}

    # emission probabilities for all states for letter A
    a_emission_probs = {  'A1'  : [0.2, 0.8],
                          'A2'  : [0.8, 0.2],
                          'A3'  : [0.2, 0.8],
                          'Aend': [0.0, 0.0]}

    """Letter Y"""
    # expected states names for letter Y
    y_states = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Yend']

    y_prior_probs = {'Y1' : 0.333,
                     'Y2' : 0.0,
                     'Y3' : 0.0,
                     'Y4' : 0.0,
                     'Y5' : 0.0,
                     'Y6' : 0.0,
                     'Y7' : 0.0,
                     'Yend' : 0.0}

    # transition probabilities between states for letter Y
    y_transition_probs = {'Y1' : {'Y1' : 0.667,
                             'Y2' : 0.333,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y2' :  {'Y1' : 0.0,
                             'Y2' : 0.2,
                             'Y3' : 0.8,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y3' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.2,
                             'Y4' : 0.8,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y4' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.2,
                             'Y5' : 0.8,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y5' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.667,
                             'Y6' : 0.333,
                             'Y7' : 0.0,
                             'Yend' : 0.0},
                    'Y6' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.2,
                             'Y7' : 0.8,
                             'Yend' : 0.0},
                    'Y7' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.667,
                             'Yend' : 0.111,
                             'L1'  : 0.111,
                             'W1'  : 0.111},
                    'Yend' :  {'Y1' : 0.0,
                             'Y2' : 0.0,
                             'Y3' : 0.0,
                             'Y4' : 0.0,
                             'Y5' : 0.0,
                             'Y6' : 0.0,
                             'Y7' : 0.0,
                             'Yend' : 1.0}}

    # emission probabilities for all states for letter Y
    y_emission_probs = { 'Y1' : [0.2, 0.8],
                         'Y2' : [0.8, 0.2],
                         'Y3' : [0.2, 0.8],
                         'Y4' : [0.8, 0.2],
                         'Y5' : [0.2, 0.8],
                         'Y6' : [0.8, 0.2],
                         'Y7' : [0.2, 0.8],
                         'Yend' : [0.0, 0.0]}    

    """Letter Z"""
    # expected states names for letter Z
    z_states = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Zend']

# prior probabilities for all states for letter Z
    z_prior_probs = {'Z1' : 0.333,
                     'Z2' : 0.0,
                     'Z3' : 0.0,
                     'Z4' : 0.0,
                     'Z5' : 0.0,
                     'Z6' : 0.0,
                     'Z7' : 0.0,
                     'Zend' : 0.0}

    # transition probabilities between states for letter Z
    z_transition_probs = {'Z1' : {'Z1' : 0.667,
                             'Z2' : 0.333,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z2' :  {'Z1' : 0.0,
                             'Z2' : 0.2,
                             'Z3' : 0.8,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z3' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.667,
                             'Z4' : 0.333,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z4' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.2,
                             'Z5' : 0.8,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z5' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.2,
                             'Z6' : 0.8,
                             'Z7' : 0.0,
                             'Zend' : 0.0},
                    'Z6' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.2,
                             'Z7' : 0.8,
                             'Zend' : 0.0},
                    'Z7' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.2,
                             'Zend' : 0.267,
                             'L1'  : 0.267,
                             'W1'  : 0.267},
                    'Zend' :  {'Z1' : 0.0,
                             'Z2' : 0.0,
                             'Z3' : 0.0,
                             'Z4' : 0.0,
                             'Z5' : 0.0,
                             'Z6' : 0.0,
                             'Z7' : 0.0,
                             'Zend' : 1.0}}

    # emission probabilities for all states for letter Z
    z_emission_probs = { 'Z1' : [0.2, 0.8],
                         'Z2' : [0.8, 0.2],
                         'Z3' : [0.2, 0.8],
                         'Z4' : [0.8, 0.2],
                         'Z5' : [0.2, 0.8],
                         'Z6' : [0.8, 0.2],
                         'Z7' : [0.2, 0.8],
                         'Zend' : [0.0, 0.0]} 

    """Pause between letters"""
    # expected states names for letter pause
    letter_pause_states = ['L1']

    # prior probabilities for all states for letter pause
    letter_pause_prior_probs = {'L1': 1.0}

    # transition probabilities between states for letter pause
    letter_pause_transition_probs = {'L1': {'L1': 0.667,
                                            'A1': 0.111,
                                            'Y1': 0.111,
                                            'Z1': 0.111}}

    # emission probabilities for all states for letter pause
    letter_pause_emission_probs = {'L1': [0.8, 0.2]}

    """Space between words"""
    # expected states names for word space
    word_space_states = ['W1']

    # prior probabilities for all states for word space
    word_space_prior_probs = {'W1': 1.0}

    # transition probabilities between states for word space
    word_space_transition_probs = {'W1': {'W1': 0.857,
                                          'A1': 0.048,
                                          'Y1': 0.048,
                                          'Z1': 0.048}}

    # emission probabilities for all states for word space
    word_space_emission_probs = {'W1': [0.8, 0.2]}

    return (a_states,
            a_prior_probs,
            a_transition_probs,
            a_emission_probs,
            y_states,
            y_prior_probs,
            y_transition_probs,
            y_emission_probs,
            z_states,
            z_prior_probs,
            z_transition_probs,
            z_emission_probs,
            letter_pause_states,
            letter_pause_prior_probs,
            letter_pause_transition_probs,
            letter_pause_emission_probs,
            word_space_states,
            word_space_prior_probs,
            word_space_transition_probs,
            word_space_emission_probs)


def quick_check() -> tuple:
    """Returns a few select values to check for accuracy.

    Returns:
        The following probabilities:
            ( prior probability of Z1,
              transition probability from Y7 to Y7,
              transition probability from Z3 to Z4,
              transition probability from W1 to W1,
              transition probability from L1 to Y1 )
    """


    #raise NotImplementedError

    (a_states,
    a_prior_probs,
    a_transition_probs,
    a_emission_probs,
    y_states,
    y_prior_probs,
    y_transition_probs,
    y_emission_probs,
    z_states,
    z_prior_probs,
    z_transition_probs,
    z_emission_probs,
    letter_pause_states,
    letter_pause_prior_probs,
    letter_pause_transition_probs,
    letter_pause_emission_probs,
    word_space_states,
    word_space_prior_probs,
    word_space_transition_probs,
    word_space_emission_probs) = part_2_a()
    #f = part_2_a()
    #print("f: ", f)
    #print("testing: ", z_prior_probs.get("Z1"))

    # prior probability for Z1
    prior_prob_Z1 = z_prior_probs.get("Z1","")  

    # transition probability from Y7 to Y7
    transition_prob_Y7_Y7 = y_transition_probs.get("Y7")["Y7"] 

    # transition probability from Z3 to Z4
    transition_prob_Z3_Z4 = z_transition_probs.get("Z3")["Z4"]  

    # transition probability from W1 to W1
    transition_prob_W1_W1 = word_space_transition_probs.get("W1")["W1"]  

    # transition probability from L1 to Y1
    transition_prob_L1_Y1 = letter_pause_transition_probs.get("L1")["Y1"]  

    return (prior_prob_Z1,
            transition_prob_Y7_Y7,
            transition_prob_Z3_Z4,
            transition_prob_W1_W1,
            transition_prob_L1_Y1)


def part_2_b(evidence_vector: list, states: list, prior_probs: dict,
            transition_probs: dict, emission_probs: dict) -> tuple:


    """Decode the most likely string generated by the evidence vector.

    Note: prior, states, transition_probs, and emission_probs will now contain
    all the letters, pauses, and spaces from part_2_a.

    For example, prior is now:

    prior_probs = {'A1'   : 0.333,
                   'A2'   : 0.0,
                   'A3'   : 0.0,
                   'Aend' : 0.0,
                   'Y1'   : 0.333,
                   'Y2'   : 0.0,
                   .
                   .
                   .
                   'Z1'   : 0.333,
                   .
                   .
                   .
                   'L1'  : 0.0,
                   'W1   : 0.0}

    Expect the same type of combinations for all probability and state input
    arguments.

    Essentially, the built Viterbi Trellis will contain all states for A, Y, Z,
    letter pause, and word space.

    Args:
        evidence_vector (list(int)): List of 0s (Silence) or 1s (Dot/Dash).
            example: [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
        states (list(string)): List of all states.
            example: ['A1', 'A2', 'A3', 'Aend']
        prior_probs (dict): prior distribution for each state.
            example: {'A1'  : 1.0,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}
        transition_probs (dict): dictionary representing transitions from
            each state to every other state, including self.
            example: {'A1'  : {'A1'  : 0.0,
                               'A2'  : 1.0,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 1.0,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}
        emission_probs (dict): dictionary of probabilities of outputs from
            each state.
            example: {'A1'  : [0.0, 1.0],
                      'A2'  : [1.0, 0.0],
                      'A3'  : [0.0, 1.0],
                      'Aend': [0.0, 0.0]}

    Returns:
        ( A string that best fits the evidence,
          probability of that string being correct as a float. )

        For example:
            an evidence vector of [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
            would return the String 'AA' with it's probability
    """

    #raise NotImplementedError

    debug = False
    sequence = ''
    probability = 0.0

    if debug:
        print("states: ", states)
        print("evidence_vector: ", evidence_vector)
        print("prior_probs: " ,prior_probs)
        print("transition_probs: ", transition_probs)
        print("emission_probs: ", emission_probs)
        print("dictionary: ", transition_probs.get(states[0],0))
        print("sample: ",transition_probs.get(states[0],0)[states[2]])
        print("answer: ",transition_probs.get(states[0],0).get(states[2],0))
        print("type: ", type(transition_probs.get(states[0],0)))

    states_len = len(states)
    input_len=len(evidence_vector)
    Trelis = np.zeros((states_len,input_len))
    #Trelis_pointer = np.chararray((states_len,input_len), itemsize=2)
    Trelis_pointer = np.zeros((states_len,input_len), dtype=int)

    for i in range(states_len):
        #Trelis[i][0]=prior_probs.get(states[i]) * emission_probs[states[i]][evidence_vector[0]]
        Trelis[i][0]=prior_probs.get(states[i],0) * emission_probs.get(states[i],0)[evidence_vector[0]]
        Trelis_pointer[i][0] = 0

    for transition in range(1, input_len):
        for i in range(states_len):
            #best_transition_prob = Trelis[0][transition-1]*transition_probs[states[0]][states[i]]
            best_transition_prob = Trelis[0][transition-1]*transition_probs.get(states[0],0).get(states[i],0)
            pointer = 0
            for z in range(1,states_len):
                #other_transition_probs = Trelis[z][transition-1]*transition_probs[states[z]][states[i]]
                other_transition_probs = Trelis[z][transition-1]*transition_probs.get(states[z],0).get(states[i],0)
                if other_transition_probs > best_transition_prob:
                    best_transition_prob = other_transition_probs
                    pointer = z

            best_prob = best_transition_prob * emission_probs.get(states[i],0)[evidence_vector[transition]]
            Trelis[i][transition] = best_prob
            Trelis_pointer[i][transition] = pointer

    if debug:
        print("\nTrelis end: \n", Trelis)
        print("\nTrelis pointer: \n", Trelis_pointer)
        print("pointer 2, 4: ", Trelis_pointer[2][4])


    # The highest probability
    sequence=[]
    probability = np.amax(Trelis, axis=0)[-1]
    backtrack = np.argmax(Trelis, axis=0)[-1]
    backtrack_state = states[backtrack]
    sequence.append(backtrack_state)

    for t in range(input_len - 1, 0, -1):
        backtrack = Trelis_pointer[backtrack][t]
        backtrack_state=states[backtrack]
        sequence.insert(0,backtrack_state)


    # change sequence to readable
    if debug:
        print("sequence: ", sequence)
    result = []
    for state in sequence:
        result.append(state[0])

    final_sequence = []
    old_letter = result[0][0]
    for letter in result:
        if letter == old_letter:
            pass
        else: 
            final_sequence.append(old_letter)
            old_letter = letter
    final_sequence.append(result[-1][0])
    #print("final sequence: ", final_sequence)

    final_seq = []
    for letter in final_sequence:
        if letter == 'W':
            letter = " "
            final_seq.append(letter)
        elif letter == 'L':
            pass
        else: 
            final_seq.append(letter)


    seperator = ''
    sequence = seperator.join(final_seq)

    if debug:
        print()
        print("______________ new ____________")
        print("sequence: ", sequence)
        print("probability: ", probability)
        print("____________ end ______________")
        print()

    return sequence, probability
