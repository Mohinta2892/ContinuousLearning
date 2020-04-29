import pandas as pd
import csv

from SI import SI
from DQN import DQN
from utilities import *
import torch.nn.functional as F
from environments.Trials import TrialEnv


scenarios = [
    # EGO-CENTRIC
    {"GOAL": TrialEnv.EAST,  "AGENT_POS": TrialEnv.SOUTH},  # 0
    {"GOAL": TrialEnv.WEST,  "AGENT_POS": TrialEnv.NORTH},  # 1
    {"GOAL": TrialEnv.EAST,  "AGENT_POS": TrialEnv.NORTH},  # 2
    {"GOAL": TrialEnv.WEST,  "AGENT_POS": TrialEnv.SOUTH},  # 3

    # ALLO-CENTRIC
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
        if stradegy == 1: return 5
        if stradegy == 2: return 6
        if stradegy == 3: return 1
        if stradegy == 4: return 2
    
    else:
        if stradegy == 1: return 4
        if stradegy == 2: return 7
        if stradegy == 3: return 0
        if stradegy == 4: return 3
        

    # wrong config
    return -1


# GENERATE MAZES
environments = []
for scenario in scenarios:
    env = TrialEnv(scenario["AGENT_POS"], scenario["GOAL"])
    environments.append(env)

NAME = "SC03"

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
            ewc_importance=800 )


activations = []
def get_activation():
    def hook(model, input, output):
        if(len(output) == 1):
            activations.append(F.relu(output.detach()[0]))
    return hook

# dqn.eval_model.fc1.register_forward_hook(get_activation())

# STRADEGY
# 1/2 - ALLOCENTRIC
# 3/4 - EGOCENTRIC

ego = []
allo = []
last_strategy = 3   # First from sc03...
trial_outcomes = []
activation_track = []
ewc = None
si = None
total_steps = 0

total = 0
show = False

for trial in trials:
    stradegy = trial["Strategy"]
    start_zone = trial["Start zone"]
    activations = []

    config = get_maze_config(stradegy, start_zone)
    env = environments[config]

    if(last_strategy != stradegy and stradegy != 5):
        print(f"Changing strategy from {last_strategy} to {stradegy}")
        last_strategy = stradegy
        dqn.reset_training()

        # ewc = EWC(dqn)

        # if si is None: 
        #     si = SI(dqn, 0.5, 0.1)
        #     si.refresh_W()
            
        # si.update_omega()
        # si.refresh_W()

        

    steps = runDQN(dqn, ewc, si, env, show)
    total_steps += steps

    finished = steps < 200
    if finished: total += 1;
    outcome = {"Trial": trial['Trial'], "Finished": finished}

    # TRACK ACTIVATIONS FOR MIDDLE LAYER
    # for timestep, activation in enumerate(activations):
    #     for ind, neuron in enumerate(activation):
    #         track = {"Trial": trial["Trial"], "Strategy": stradegy, "Timestep": timestep, "Neuron": ind+1, "Activation": neuron.item()}
    #         activation_track.append(track)

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
print(f"Average steps: {total_steps/len(trials)}")
# np.save("test2", np.array(activation_track))
np.save(f"data/trials/{NAME}_SI_allo_3", np.array(allo))
np.save(f"data/trials/{NAME}_SI_ego_3", np.array(ego))

print("Done saving 1")
# SAVE OUTPUT
# with open('trial_finished.txt', 'w', newline='') as output_file:
#     keys = trial_outcomes[0].keys()
#     dict_writer = csv.DictWriter(output_file, keys)
#     dict_writer.writeheader()
#     dict_writer.writerows(trial_outcomes)

# with open(f"{NAME}_activations.csv", 'w', newline='') as output_file:
#     keys = activation_track[0].keys()
#     dict_writer = csv.DictWriter(output_file, keys)
#     dict_writer.writeheader()
#     dict_writer.writerows(activation_track)


print("Done saving 2")


    

