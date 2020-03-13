import pandas as pd
from DQN import DQN
from utilities import *
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

    # START ZONE
    # 1 - NORTH
    # 2 - SOUTH

    # STRADEGY
    # 1/2 - ALLOCENTRIC
    # 3/4 - EGOCENTRIC

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


# GENERATE MAZES
environments = []
for scenario in scenarios:
    env = TShapedEnv(scenario["AGENT_POS"], scenario["GOAL"])
    environments.append(env)

# READ TRIALS
trials = pd.read_csv(csv_path).T.to_dict().values()

# GET DQN NET
env_action_num = 3
env_states_num = environments[0].getStateSize()
dqn = DQN(  gamma=0.9,
            memory_size=10_000,
            target_update_counter=25,
            batch_size=32,
            num_of_states=env_states_num,
            num_of_actions=env_action_num,
            ewc_importance=1000 )


# RUN TRIALS
seen_environments = set()
trial_steps = []
for trial in trials:
    stradegy = trial["Strategy"]
    start_zone = trial["Start zone"]

    config = get_maze_config(stradegy, start_zone)
    env = environments[config]

    if not env in seen_environments:
        train(dqn, env, EPISODES=200, DISPLAY_FREQUENCY=50)

    test_steps = test(dqn.eval_model, env, should_render=False)
    trial_steps.append(test_steps)
    print(f"TRIAL STEPS: {trial_steps}")

    seen_environments.add(env)




    

