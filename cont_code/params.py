''' File that sets up the parameters for the simulation to be run.
    This is the file that controls the distribution of labeled datapoints
    that both algorithms see, and also the action set with which EXP3 is run.
'''

import numpy as np
import random
import math
from copy import deepcopy
from grinding_polytopes import * 
import polytope as pc
import logging
logging.basicConfig()

def set_params(T, eps_exp3): 
    np.random.seed()
    d = 2 # dimension of space
    p = 0.7 
    # variable is controlled by the outside file
    delta = 0.05 #radius of best-responses -- implicitly affects regret
    agent_type = [0]*T

    
    agent_type = np.random.binomial(1,p,T)

    true_labels = [1 if agent_type[i] else -1 for i in range(T)] 

    #original feature vectors for agents
    x_real = []
    for i in range(T):
        if agent_type[i]:
            x_real.append(np.array([np.random.normal(0.6, 0.4), np.random.normal(0.4,0.6), 1]))
        else:
            x_real.append(np.array([np.random.normal(0.4, 0.6), np.random.uniform(0.6,0.4), 1]))

    calA_size_exp3   = 1000

    noise = []

    initial = []
    zero = np.array([0, 0, 1])
    one  = np.array([1, 1, 1])
    curr_size = 0

    while curr_size < calA_size_exp3:
        temp  = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)])
        dist0 = np.abs(1.0*np.dot(temp,zero)/np.linalg.norm(temp[:d]))  
        dist1 = np.abs(1.0*np.dot(temp,one)/np.linalg.norm(temp[:d]))  
        if dist0 <= np.sqrt(2) and dist1 <= np.sqrt(2):
            initial.append(temp)
            curr_size += 1


    calA_size = len(initial)

    # construct initial polytope, i.e., [-1,1]^{d+1}
    V = np.array([  np.array([-1, -1, -1]), 
                    np.array([-1, -1,  1]),
                    np.array([-1,  1, -1]),
                    np.array([-1,  1,  1]),
                    np.array([ 1, -1, -1]),
                    np.array([ 1, -1,  1]),
                    np.array([ 1,  1, -1]),
                    np.array([ 1,  1,  1])])

    p_init = pc.qhull(V)

    # start with a prob and weight of 1 for the org polytope
    calA_exp3        = [init/np.linalg.norm(init[:d]) for init in initial]
    updated = [0]*T
    initial_polytope = Grind_Polytope(p_init, 1.0, 1.0, 2, T, 0.0, 0.0, updated)
    calA_grind = [initial_polytope] 

    return (T, d, x_real, calA_exp3, calA_grind, agent_type, true_labels, delta, noise, p)
    

