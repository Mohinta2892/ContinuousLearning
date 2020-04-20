import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

NAME = "sc03"
csv_path = f"data/trials/{NAME}.csv"

allo_cnt = np.load(f"data/trials/{NAME}_allo.npy", allow_pickle=True)
ego_cnt = np.load(f"data/trials/{NAME}_ego.npy", allow_pickle=True)
trials = pd.read_csv(csv_path).T.to_dict().values()

# Unpack
# allo_cnt = [x[1] for x in allo_cnt]
# ego_cnt = [x[1] for x in ego_cnt]

sections = []
last_task = -1
last_strategy = -1
strategy_switches = []


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

   
fig=plt.figure(figsize=[30,5])
ax=fig.add_subplot(111)
   
for offset, section in enumerate(sections):
   isAllo = section[0] == 0
   Xfrom  = section[1]
   Xto    = section[-1]

   Y = allo_cnt if isAllo else ego_cnt
   Y = [x[1] for x in Y]
   Y_avg = movingaverage(Y[Xfrom-1-offset : Xto-1-offset], 6)

   color = 'r' if isAllo else 'b'

   ax.plot(np.arange(Xfrom, Xto), Y[Xfrom-1-offset : Xto-1-offset], color=color, linestyle='--', linewidth=0.5, label= "Allocentric" if isAllo else "Egocentric")
   ax.plot(np.arange(Xfrom, Xto), Y_avg, color=color, linewidth=2, label= "Allocentric Average" if isAllo else "Egocentric Average")

for section in sections:
   X = section[-1]
   ax.axvline(X, color='purple', linestyle='--', linewidth=1, label="Task Switch")

for switch in strategy_switches:
   ax.axvline(switch, color='green', linestyle='--', linewidth=1, label="Within task switch")

ax.hlines(0.5,1,542,linestyles='dashed',linewidth=1, color='k', label='Chance Level')

ax.set_ylim((-0.1,1.1))
ax.set_xticks(np.arange(1, sections[-1][-1], 25))
ax.tick_params(direction='in')
ax.set_xlabel('Trials',fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Accuracy',fontsize=14, fontweight='bold')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend_properties = {'weight':'bold','size': 12}
lgd=ax.legend(by_label.values(), by_label.keys(),loc='lower center', bbox_to_anchor=(0.5, -0.24), ncol=8, fancybox=True, shadow=True, prop=legend_properties)

plt.savefig('test', bbox_inches='tight', dpi=200)
plt.show()	






y_av_110=movingaverage(reward_val[50:112], 6)
y_av_386=movingaverage(reward_val[212:388],6)
y_av_486=movingaverage(reward_val[414:488],6)

ax.plot(np.arange(50,112,1),y_av_110, color='r',linestyle='solid', linewidth=2,label='Moving Avg. (Allocentric)')
ax.plot(np.arange(212,388,1),y_av_386, color='r',linestyle='solid', linewidth=2,)
ax.plot(np.arange(414,488,1),y_av_486, color='r',)





# ax.plot(np.arange(251,500,1),reward_val_ego1[251:],color='brown',linestyle='--',linewidth=0.4, label='Ego South')

# y_av_ego_49=movingaverage(reward_val_ego1[251:],6)
# ax.plot(np.arange(251,500,1),y_av_ego_49, color='brown',linestyle='solid', linewidth=2,label='Moving Avg. (Ego South)')


ax.plot(np.arange(1,51,1),reward_val_ego1[:50],color='b',linestyle='--',linewidth=0.4, label='Egocentric')
ax.plot(np.arange(112,213,1),reward_val_ego1[112:213],color='b',linestyle='--',linewidth=0.4,)
ax.plot(np.arange(388,415,1),reward_val_ego1[388:415],color='b',linestyle='--',linewidth=0.4,)
ax.plot(np.arange(488,542,1),reward_val_ego1[488:],color='b',linestyle='--',linewidth=0.4,)


y_av_ego_49=movingaverage(reward_val_ego1[:51],6)
y_av_ego_211=movingaverage(reward_val_ego1[112:213],6)
y_av_ego_413=movingaverage(reward_val_ego1[388:415],6)
y_av_ego_487=movingaverage(reward_val_ego1[488:],6)

ax.plot(np.arange(1,52,1),y_av_ego_49, color='b',linestyle='solid', linewidth=2,label='Moving Avg. (Egocentric)')

ax.plot(np.arange(112,213,1),y_av_ego_211, color='b',linestyle='solid', linewidth=2,)
ax.plot(np.arange(388,415,1),y_av_ego_413, color='b',linestyle='solid', linewidth=2,)
ax.plot(np.arange(488,542,1),y_av_ego_487, color='b',linestyle='solid', linewidth=2,)



    
ax.set_xticks(np.arange(1, 525,25))
ax.set_xlabel('Trials',fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Accuracy',fontsize=14, fontweight='bold')
ax.set_ylim((-0.1,1.1))
fig.suptitle("Continual learning - only allocentric with switching start locations",fontsize=16, fontweight='bold')
ax.hlines(0.5,1,542,linestyles='dashed',linewidth=1, color='k', label='Chance Level')

'''Target lines : mean performance'''
target_avg_allo=(sum(reward_val1[1:250]))/250.0
target_avg_ego=(sum(reward_val_ego1[251:501]))/250.0

ax.hlines(target_avg_allo,1,500,linestyles='-.',linewidth=2.2, color='g', label='Allocentric North Perf. Mean')
ax.hlines(target_avg_ego,1,500,linestyles='-.',linewidth=2.2,color='brown', label='Allocentric South Perf. Mean')

handles, labels = ax.get_legend_handles_labels()
ax.tick_params(direction='in')

lgd=ax.legend(handles, labels,loc='lower center', bbox_to_anchor=(0.5, -0.24), ncol=8, fancybox=True, shadow=True,prop=legend_properties)


plt.savefig('test/only_ego_rnn', bbox_inches='tight', dpi=200)
plt.show()