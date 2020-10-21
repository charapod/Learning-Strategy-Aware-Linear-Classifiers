''' Driver file '''

from principal_exp3 import *
from principal_grind import *
from agent import *
from oracle_cont import *
from copy import deepcopy
from master_file import regret_grind, regret_exp3
from params import set_params
import math

num_repetitions = 30
dgrind  = [] 
exp3    = []
min_num_rounds = 0
max_num_rounds = 1000
step = 5
rounds = [T for T in range(min_num_rounds,max_num_rounds)] 

T = max_num_rounds
(num_agents, dim, x_real, calA_exp3, calA_grind, agent_types, true_labels, delta, noise, prob) = set_params(T, 0.2)

cp_xreal       = deepcopy(x_real)
for delta in [0.05,0.1, 0.15, 0.3, 0.5]:
    print ("Current delta = %.5f"%delta)
    agents_grind   = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]
    oracle_grind   = Oracle(deepcopy(agents_grind), T) 

    principal_grind = [Principal_Grind(T, calA_grind, num_repetitions) for _ in range(0, num_repetitions)] 
    principal_exp3  = [Principal_Exp3(T, calA_exp3, num_repetitions, 0) for _ in range(0, num_repetitions)] 

    agents_exp3     = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]


    oracle_exp3     = Oracle(deepcopy(agents_exp3), T) 
    resp_lst_exp3   = oracle_exp3.compute_responses(deepcopy(calA_exp3), dim)

    (exp3, exp3_regrets, best_fixed) = regret_exp3(principal_exp3, agents_exp3, oracle_exp3, resp_lst_exp3, T, num_repetitions, num_agents, dim)  
    best_fixed = [[0]*T for _ in range(num_repetitions)]
    (grind, grind_regrets) = regret_grind(1, principal_grind, agents_grind, oracle_grind, T, num_repetitions, num_agents, dim, best_fixed, prob)  

