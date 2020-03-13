import pandas as pd
from DQN import DQN
from environments.TShapedEnv import TShapedEnv

csv_path = 'environments/configs.csv'

scenarios = [
    # ALLO-CENTRIC
    {"GOAL": TShapedEnv.EAST,  "AGENT_POS": TShapedEnv.SOUTH},
    {"GOAL": TShapedEnv.WEST,  "AGENT_POS": TShapedEnv.NORTH},
    {"GOAL": TShapedEnv.EAST,  "AGENT_POS": TShapedEnv.NORTH},
    {"GOAL": TShapedEnv.WEST,  "AGENT_POS": TShapedEnv.SOUTH},

    # EGO-CENTRIC
    {"GOAL": TShapedEnv.WEST, "AGENT_POS": TShapedEnv.SOUTH},
    {"GOAL": TShapedEnv.WEST, "AGENT_POS": TShapedEnv.NORTH},
    {"GOAL": TShapedEnv.EAST, "AGENT_POS": TShapedEnv.NORTH},
    {"GOAL": TShapedEnv.EAST, "AGENT_POS": TShapedEnv.SOUTH}
]

def run(env):

    state = env.reset()
    state = env.extractState()
    steps_taken = 0
    isFinished = False

    while not isFinished:
        env.render()

    return steps_taken

def get_maze_config(stradegy, start_zone):

    if start_zone == 1:
        if stradegy == 1: return 1
        if stradegy == 2: return 2
        if stradegy == 3: return 5
        if stradegy == 4: return 6
    
    else:
        if stradegy == 1: return 0
        if stradegy == 2: return 3
        if stradegy == 3: return 4
        if stradegy == 4: return 7

    # wrong config
    return -1


# START ZONE
# 1 - NORTH
# 2 - SOUTH

# STRADEGY
# 1/2 - ALLOCENTRIC
# 3/4 - EGOCENTRIC

# GENERATE MAZES
environments = []
for scenario in scenarios:
    env = TShapedEnv(scenario["AGENT_POS"], scenario["GOAL"])
    environments.append(env)

# READ TRIALS
trials = pd.read_csv(csv_path).T.to_dict().values()

# GET DQN NET
dqn = DQN(GAMMA, MEMORY_SIZE, TARGET_UPDATE, BATCH_SIZE, env_state_num, env_action_num, ewc_importance=1500)


# RUN TRIALS
for trial in trials:
    stradegy = trial["Strategy"]
    start_zone = trial["Start zone"]

    config = get_maze_config(stradegy, start_zone)
    env = environments[config]



    

