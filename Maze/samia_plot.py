import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics 

def extractRat(trials):
   ego = []
   allo = []
   last_strategy = 3   # First from sc03...
   trial_outcomes = []

   total = 0

   for trial in trials:
      stradegy = trial["Strategy"]
      start_zone = trial["Start zone"]
      finished = trial["Correct"]

      if(last_strategy != stradegy and stradegy != 5):
         last_strategy = stradegy

      if(stradegy == 1 or stradegy == 2):
         allo.append([trial['Trial'], int(finished)])
         ego.append([trial['Trial'], 0])
      else:
         ego.append([trial['Trial'], int(finished)])
         allo.append([trial['Trial'], 0])

   return allo, ego

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def getTrialsPlot(NAME, EWC, SI):

   path = "no_EWC"
   if EWC: 
      path = "EWC"
   if SI:
      path = "SI"

   csv_path = f"data/trials/{NAME}.csv"

   trials = pd.read_csv(csv_path).T.to_dict().values()
   allo_cnt = np.load(f"data/trials/{NAME}_{path}_allo.npy", allow_pickle=True)
   ego_cnt = np.load(f"data/trials/{NAME}_{path}_ego.npy", allow_pickle=True)
   # allo_cnt, ego_cnt = extractRat(trials)

   sections = []
   last_task = -1
   last_strategy = -1
   strategy_switches = []
   total_ego = 0
   total_allo = 0
   correct_ego = 0
   correct_allo = 0

   for trial in trials:
      trialID = trial["Trial"]
      stradegyID = trial["Strategy"]      

      stradegy = 0 if (stradegyID == 1 or stradegyID == 2) else 1

      if last_task == -1 or last_task != stradegy:
         #  new stradegy => open new section
         last_task = stradegy
         sections.append([])
         sections[-1].append(stradegy)
         last_strategy = stradegyID
      elif last_strategy != stradegyID:
         strategy_switches.append(trialID)
         last_strategy = stradegyID

      sections[-1].append(trialID)
      if stradegy == 0:
         total_allo += 1
         if allo_cnt[trialID-1-len(sections)+1][1] == 1: correct_allo += 1
      else:
         total_ego += 1
         if ego_cnt[trialID-1-len(sections)+1][1] == 1: correct_ego += 1

      
   fig=plt.figure(figsize=[20,6])
   ax=fig.add_subplot(111)
      
   for offset, section in enumerate(sections):
      isAllo = section[0] == 0
      Xfrom  = section[1]
      Xto    = section[-1]

      Y = allo_cnt if isAllo else ego_cnt
      Y = [x[1] for x in Y]
      Y_avg = movingaverage(Y[Xfrom-1-offset : Xto-1-offset], 6)

      color = 'r' if isAllo else 'b'

      # ax.plot(np.arange(Xfrom, Xto), Y[Xfrom-1-offset : Xto-1-offset], color=color, linestyle='--', linewidth=0.5, label= "Allocentric" if isAllo else "Egocentric")
      ax.plot(np.arange(Xfrom, Xto), Y_avg, color=color, linewidth=2, label= "Allocentric Avg." if isAllo else "Egocentric Avg.")

   for section in sections:
      X = section[-1]
      ax.axvline(X, color='purple', linestyle='--', linewidth=1.2, label="Task Switch")

   for switch in strategy_switches:
      ax.axvline(switch, color='green', linestyle='--', linewidth=1.2, label="Within Task Switch")

   ax.hlines(0.5,1,len(trials),linestyles='dashed',linewidth=1.2, color='k', label='Chance Level')
   ax.hlines(correct_allo/total_allo,1,len(trials),linestyles='dashed',linewidth=1.2, color='r', label='Allocentric Perf. Mean')
   ax.hlines(correct_ego/total_ego,1,len(trials),linestyles='dashed',linewidth=1.2, color='b', label='Egocentric Perf. Mean')

   print("Allo perf: ", correct_allo/total_allo);
   print("Ego perf: ", correct_ego/total_ego);

   ax.set_ylim((-0.1,1.1))
   ax.set_xticks(np.arange(1, sections[-1][-1], 25))
   ax.tick_params(size=9)
   ax.set_xlabel('Trials',fontsize=14, fontweight='bold')
   ax.set_ylabel('Performance Accuracy',fontsize=14, fontweight='bold')

   handles, labels = ax.get_legend_handles_labels()
   by_label = dict(zip(labels, handles))
   legend_properties = {'weight':'bold','size': 13}
   lgd=ax.legend(by_label.values(), by_label.keys(),loc='lower center', bbox_to_anchor=(0.5, -0.24), ncol=7, fancybox=True, shadow=True, prop=legend_properties)

   extention = ""
   if EWC: extention = "-EWC"
   if SI: extention = "-SI"
   fig.suptitle(f"Continual learning across allocentric and egocentric tasks with DQN{extention}",fontsize=16, fontweight='bold')
   # fig.suptitle(f"Continual learning across allocentric and egocentric tasks from Mice",fontsize=16, fontweight='bold')
   plt.margins(0.01,0.01)
   plt.savefig(f"plots/Trials-{NAME}-{path}", bbox_inches='tight', dpi=200)
   # plt.savefig(f"plots/Trials-{NAME}-{path}EWC.pdf", bbox_inches='tight', dpi=200)
   # plt.savefig(f"plots/Trials-{NAME}-Mice", bbox_inches='tight', dpi=200)
   plt.show()	

def getAcc(allo, elo, trials):
   total_allo = 0
   total_ego = 0
   correct_allo = 0
   correct_ego = 0
   last_task = -1
   sections = 0

   for trial in trials:
      trialID = trial["Trial"]
      stradegyID = trial["Strategy"]      

      stradegy = 0 if (stradegyID == 1 or stradegyID == 2) else 1

      if last_task == -1 or last_task != stradegy:
         #  new stradegy => open new section
         last_task = stradegy
         last_strategy = stradegyID
         sections += 1
      elif last_strategy != stradegyID:
         last_strategy = stradegyID

      if stradegy == 0:
         total_allo += 1
         if allo[trialID-1-sections+1][1] == 1: correct_allo += 1
      else:
         total_ego += 1
         if elo[trialID-1-sections+1][1] == 1: correct_ego += 1

   pallo = correct_allo/total_allo
   pego = correct_ego/total_ego

   return statistics.mean([pallo, pego]), statistics.stdev([pallo, pego])
   # return correct_allo + correct_ego


def getAccPlot(NAME):
   csv_path = f"data/trials/{NAME}.csv"

   trials = pd.read_csv(csv_path).T.to_dict().values()
   allo = np.load(f"data/trials/{NAME}_no_EWC_allo.npy", allow_pickle=True)
   ego = np.load(f"data/trials/{NAME}_no_EWC_ego.npy", allow_pickle=True)
   ewc_allo = np.load(f"data/trials/{NAME}_EWC_allo.npy", allow_pickle=True)
   ewc_ego = np.load(f"data/trials/{NAME}_EWC_ego.npy", allow_pickle=True)
   si_allo = np.load(f"data/trials/{NAME}_SI_allo_3.npy", allow_pickle=True)
   si_ego = np.load(f"data/trials/{NAME}_SI_ego_3.npy", allow_pickle=True)
   rat_allo, rat_ego = extractRat(trials)

   acc_rat, err_rat = getAcc(rat_allo, rat_ego, trials)
   acc_dqn, err_dqn = getAcc(allo, ego, trials)
   acc_ewc, err_ewc = getAcc(ewc_allo, ewc_ego, trials)
   acc_si, err_si = getAcc(ewc_allo, ewc_ego, trials)

   fig=plt.figure(figsize=[9,6])
   ax=fig.add_subplot(111)

   index = np.arange(4)
   ax.set_xlabel('Models', fontsize=14, fontweight='bold')
   ax.set_ylabel('Performance accuracy', fontsize=14, fontweight='bold')
   ax.set_xticks(index)
   ax.tick_params(axis='both', which='major', labelsize=12)
   ax.set_xticklabels(["Mice", "DQN", "DQN-EWC", "DQN-SI"])
   ax.set_yticks(np.arange(0, 1.1, 0.2))
   ax.set_ylim((-0.1,1.1))
   # ax.title('Market Share for Each Genre 1995-2017)

   ax.hlines(0.5, -0.5, len(index)-0.5,  linestyle='dashed',linewidth=1.2, color='k', label='Chance Level')
   # plt.axhline(y=0.5, linestyle='dashed',linewidth=1.2, color='k', label='Chance Level')
   # plt.margins(0.01,0.01)
   
   ax.bar(index, [acc_rat, acc_dqn, acc_ewc, acc_si], yerr=[err_rat, err_dqn, err_ewc, err_si], color=['#FFD17F', '#7FBF7F', '#FF7F7F', '#92E5FF'], align='center', alpha=0.85, ecolor='black', capsize=10)

   legend_properties = {'weight':'bold','size': 13}
   ax.legend(prop=legend_properties, loc="upper left")

   plt.savefig(f"plots/Trials-{NAME}-Accuracies", bbox_inches='tight', dpi=200)
   plt.show()	

def getHiddenAcc(NAME):
   csv_path = f"data/trials/{NAME}.csv"
   trials = pd.read_csv(csv_path).T.to_dict().values()
   total = len(trials)

   allo821 = np.load(f"data/trials/{NAME}_no_EWC_allo_82_1.npy", allow_pickle=True)
   ego821 = np.load(f"data/trials/{NAME}_no_EWC_ego_82_1.npy", allow_pickle=True)
   allo822 = np.load(f"data/trials/{NAME}_no_EWC_allo_82_2.npy", allow_pickle=True)
   ego822 = np.load(f"data/trials/{NAME}_no_EWC_ego_82_2.npy", allow_pickle=True)
   allo823 = np.load(f"data/trials/{NAME}_no_EWC_allo_82_3.npy", allow_pickle=True)
   ego823 = np.load(f"data/trials/{NAME}_no_EWC_ego_82_3.npy", allow_pickle=True)
   c1 = getAcc(allo821, ego821, trials)
   c2 = getAcc(allo822, ego822, trials)
   c3 = getAcc(allo823, ego823, trials)
   mean82 = statistics.mean([c1, c2, c3])
   var82 = statistics.stdev([c1/total, c2/total, c3/total])

   allo1401 = np.load(f"data/trials/{NAME}_no_EWC_allo_140_1.npy", allow_pickle=True)
   ego1401 = np.load(f"data/trials/{NAME}_no_EWC_ego_140_1.npy", allow_pickle=True)
   allo1402 = np.load(f"data/trials/{NAME}_no_EWC_allo_140_2.npy", allow_pickle=True)
   ego1402 = np.load(f"data/trials/{NAME}_no_EWC_ego_140_2.npy", allow_pickle=True)
   allo1403 = np.load(f"data/trials/{NAME}_no_EWC_allo_140_3.npy", allow_pickle=True)
   ego1403 = np.load(f"data/trials/{NAME}_no_EWC_ego_140_3.npy", allow_pickle=True)
   c1 = getAcc(allo1401, ego1401, trials)
   c2 = getAcc(allo1402, ego1402, trials)
   c3 = getAcc(allo1403, ego1403, trials)
   mean140 = statistics.mean([c1, c2, c3])
   var140 = statistics.stdev([c1/total, c2/total, c3/total])

   allo2001 = np.load(f"data/trials/{NAME}_no_EWC_allo_200_1.npy", allow_pickle=True)
   ego2001 = np.load(f"data/trials/{NAME}_no_EWC_ego_200_1.npy", allow_pickle=True)
   allo2002 = np.load(f"data/trials/{NAME}_no_EWC_allo_200_2.npy", allow_pickle=True)
   ego2002 = np.load(f"data/trials/{NAME}_no_EWC_ego_200_2.npy", allow_pickle=True)
   allo2003 = np.load(f"data/trials/{NAME}_no_EWC_allo_200_3.npy", allow_pickle=True)
   ego2003 = np.load(f"data/trials/{NAME}_no_EWC_ego_200_3.npy", allow_pickle=True)
   c1 = getAcc(allo2001, ego2001, trials)
   c2 = getAcc(allo2002, ego2002, trials)
   c3 = getAcc(allo2003, ego2003, trials)
   mean200 = statistics.mean([c1, c2, c3])
   var200 = statistics.stdev([c1/total, c2/total, c3/total])

   allo3001 = np.load(f"data/trials/{NAME}_no_EWC_allo_300_1.npy", allow_pickle=True)
   ego3001 = np.load(f"data/trials/{NAME}_no_EWC_ego_300_1.npy", allow_pickle=True)
   allo3002 = np.load(f"data/trials/{NAME}_no_EWC_allo_300_2.npy", allow_pickle=True)
   ego3002 = np.load(f"data/trials/{NAME}_no_EWC_ego_300_2.npy", allow_pickle=True)
   allo3003 = np.load(f"data/trials/{NAME}_no_EWC_allo_300_3.npy", allow_pickle=True)
   ego3003 = np.load(f"data/trials/{NAME}_no_EWC_ego_300_3.npy", allow_pickle=True)
   c1 = getAcc(allo3001, ego3001, trials)
   c2 = getAcc(allo3002, ego3002, trials)
   c3 = getAcc(allo3003, ego3003, trials)
   mean300 = statistics.mean([c1, c2, c3])
   var300 = statistics.stdev([c1/total, c2/total, c3/total])

   fig=plt.figure(figsize=[9,6])
   ax=fig.add_subplot(111)

   index = np.arange(4)
   ax.set_xlabel('Hidden Width', fontsize=14, fontweight='bold')
   ax.set_ylabel('Performance accuracy', fontsize=14, fontweight='bold')
   ax.set_xticks(index)
   ax.tick_params(axis='both', which='major', labelsize=12)
   ax.set_xticklabels(["82", "140", "200", "300"])
   ax.set_yticks(np.arange(0, 1.1, 0.2))
   ax.set_ylim((-0.1,1.1))
   # ax.title('Market Share for Each Genre 1995-2017)

   ax.hlines(0.5, -0.5, len(index)-0.5,  linestyle='dashed',linewidth=1.2, color='k', label='Chance Level')
   # plt.axhline(y=0.5, linestyle='dashed',linewidth=1.2, color='k', label='Chance Level')
   # plt.margins(0.01,0.01)
   
   ax.bar(index, [mean82/total, mean140/total, mean200/total, mean300/total], yerr=[var82, var140, var200, var300], color=['#FFD17F', '#7FBF7F', '#FF7F7F', '#92E5FF'], align='center', alpha=0.85, ecolor='black', capsize=10)

   legend_properties = {'weight':'bold','size': 13}
   ax.legend(prop=legend_properties, loc="upper left")

   plt.savefig(f"plots/Trials-{NAME}-Hidden", bbox_inches='tight', dpi=200)
   plt.show()	

def getStepsPlot(NAME):

   dqn_1 = 37.548507462686565
   dqn_2 = 32.095149253731343
   dqn_3 = 31.345149253731343
   dqn_4 = 46.28917910447761
   dqn_mean = statistics.mean([dqn_1, dqn_2, dqn_3, dqn_4])
   dqn_var  = statistics.stdev([dqn_1, dqn_2, dqn_3, dqn_4])

   ewc_1 = 27.167910447761194
   ewc_2 = 21.94402985074627
   ewc_3 = 32.94589552238806
   ewc_4 = 21.78171641791045
   ewc_mean = statistics.mean([ewc_1, ewc_2, ewc_3, ewc_4])
   ewc_var  = statistics.stdev([ewc_1, ewc_2, ewc_3, ewc_4])

   si_1 = 20.824626865671643
   si_2 = 24.938432835820894
   si_3 = 15.830223880597014
   si_4 = 19.71455223880597
   si_mean = statistics.mean([si_1, si_2, si_3, si_4])
   si_var  = statistics.stdev([si_1, si_2, si_3, si_4])

   fig=plt.figure(figsize=[9,6])
   ax=fig.add_subplot(111)

   index = np.arange(3)
   ax.set_xlabel('Models', fontsize=14, fontweight='bold')
   ax.set_ylabel('Avg. Steps per trial', fontsize=14, fontweight='bold')
   ax.set_xticks(index)
   ax.tick_params(axis='both', which='major', labelsize=12)
   ax.set_xticklabels(["DQN", "DQN-EWC", "DQN-SI"])
   ax.set_yticks(np.arange(0, 50.1, 10))
   ax.set_ylim((-0.1,51))

   ax.bar(index, [dqn_mean, ewc_mean, si_mean], yerr=[dqn_var, ewc_var, si_var], color=['#FFD17F', '#7FBF7F', '#FF7F7F', '#92E5FF'], align='center', alpha=0.85, ecolor='black', capsize=10)

   plt.savefig(f"plots/Trials-{NAME}-Steps", bbox_inches='tight', dpi=200)
   plt.show()	

NAME = "sc03"
EWC = True

# getTrialsPlot(NAME, True, False)
# getAccPlot(NAME)
# getHiddenAcc(NAME)
getStepsPlot(NAME)