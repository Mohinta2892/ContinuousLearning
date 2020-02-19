import numpy as np 
import matplotlib.pyplot as plt

xs = [0, 1]
ys = [1, 0]

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)
# ax1.title.set_text('Loss')
ax1.set_xlabel('Data')
ax1.set_ylabel('Assumptions')

ax1.plot(xs, ys)
ax1.fill_between(xs, ys, color='#539ecd')


plt.savefig('data_assumption.png')

