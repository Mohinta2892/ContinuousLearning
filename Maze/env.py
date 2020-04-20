import pandas as pd
import csv

from DQN import DQN
from utilities import *
from environments.Trials import TrialEnv


scenarios = [
    # ALLO-CENTRIC
    {"GOAL": TrialEnv.EAST,  "AGENT_POS": TrialEnv.SOUTH},  # 0
    {"GOAL": TrialEnv.WEST,  "AGENT_POS": TrialEnv.NORTH},  # 1
    {"GOAL": TrialEnv.EAST,  "AGENT_POS": TrialEnv.NORTH},  # 2
    {"GOAL": TrialEnv.WEST,  "AGENT_POS": TrialEnv.SOUTH},  # 3

    # EGO-CENTRIC
    {"GOAL": TrialEnv.WEST, "AGENT_POS": TrialEnv.SOUTH},  # 4
    {"GOAL": TrialEnv.WEST, "AGENT_POS": TrialEnv.NORTH},  # 5
    {"GOAL": TrialEnv.EAST, "AGENT_POS": TrialEnv.NORTH},  # 6
    {"GOAL": TrialEnv.EAST, "AGENT_POS": TrialEnv.SOUTH}   # 7
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
        if stradegy == 5: return 5
    
    else:
        if stradegy == 1: return 0
        if stradegy == 2: return 3
        if stradegy == 3: return 4
        if stradegy == 4: return 7
        if stradegy == 5: return 4
        

    # wrong config
    return -1


# GENERATE MAZES
environments = []
for scenario in scenarios:
    env = TrialEnv(scenario["AGENT_POS"], scenario["GOAL"])
    environments.append(env)

NAME = "sc03"

showEnv(environments[6])

# READ TRIALS
# csv_path = 'environments/configs.csv'
csv_path = f"data/trials/{NAME}.csv"
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
trained_env = np.zeros(8)
trial_outcomes = []

# for trial in trials:
#     stradegy = trial["Strategy"]
#     start_zone = trial["Start zone"]

#     config = get_maze_config(stradegy, start_zone)
#     env = environments[config]

#     # if not env in seen_environments:
#     if trained_env[config] < 3:
#         print("#########################################################")
#         # print(f"Trial: {trial['Trial']} is a new config: {config}. Training...")
#         print(f"Trial: {trial['Trial']} is training config: {config} for the {trained_env[config]} time. Training...")
#         trainSmart(dqn, env, EPISODES=200, DISPLAY_FREQUENCY=50, config=config)

#     test_steps = test(dqn.eval_model, env, should_render=False)
#     # print(f"TRIAL STEPS: {trial_steps}")

#     # seen_environments.add(env)
#     trained_env[config] += 1
#     outcome = {"Trial": trial['Trial'], "Stradegy": config, "Steps": test_steps}
#     trial_outcomes.append(outcome)

# STRADEGY
# 1/2 - ALLOCENTRIC
# 3/4 - EGOCENTRIC

ego = []
allo = []
last_strategy = 3
ewc = None

total = 0

for trial in trials:
    stradegy = trial["Strategy"]
    start_zone = trial["Start zone"]

    config = get_maze_config(stradegy, start_zone)
    env = environments[config]

    if(last_strategy != stradegy and stradegy != 5):
        print(f"Changing strategy from {last_strategy} to {stradegy}")
        last_strategy = stradegy
        dqn.reset_training()
        ewc = EWC(dqn)

    steps = runDQN(dqn, ewc, env)

    finished = steps < 200
    if finished: total += 1;
    outcome = {"Trial": trial['Trial'], "Finished": finished}

    if(stradegy == 1 or stradegy == 2):
        allo.append([trial['Trial'], int(finished)])
        ego.append([trial['Trial'], 0])
    else:
        ego.append([trial['Trial'], int(finished)])
        allo.append([trial['Trial'], 0])


    print(f"Trial {trial['Trial']} ended with {finished} - {stradegy} / {start_zone}")
    trial_outcomes.append(outcome)


print(len(allo))
print(len(ego))
print(total)
np.save(f"data/trials/{NAME}_allo", np.array(allo))
np.save(f"data/trials/{NAME}_ego", np.array(ego))

# SAVE OUTPUT
with open('trial_finished.txt', 'w', newline='') as output_file:
    keys = trial_outcomes[0].keys()
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(trial_outcomes)




    

