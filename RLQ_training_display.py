# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 01:19:48 2020

@author: William Woodall
"""

from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns

import numpy as np
import pickle

train_stats_dir = './stats/training_hist.pkl'
q_val_dir = './stats/q_val.pkl'

plt.style.use('fivethirtyeight')
sns.set(style='darkgrid', palette='bright', font_scale=0.9)

fig = plt.figure(figsize=(20,12))
gs = fig.add_gridspec(13,8) 
ax = fig.add_subplot(gs[:3,:])
ax2 = fig.add_subplot(gs[4:8,:3])
ax5 = fig.add_subplot(gs[4:8,3:6])
ax7 = fig.add_subplot(gs[8:12,3:6])
ax3 = fig.add_subplot(gs[8:10,:3])
ax4 = fig.add_subplot(gs[10:12,:3])

qx1 = fig.add_subplot(gs[4:5,6:])
qx2 = fig.add_subplot(gs[5:6,6:])
qx3 = fig.add_subplot(gs[6:7,6:])
qx4 = fig.add_subplot(gs[7:8,6:])
qx5 = fig.add_subplot(gs[8:9,6:])
qx6 = fig.add_subplot(gs[9:10,6:])
qx7 = fig.add_subplot(gs[10:11,6:])
qx8 = fig.add_subplot(gs[11:12,6:])
qx9 = fig.add_subplot(gs[12:13,6:])
       
def plot_stats_animation(i):
    
    with open(train_stats_dir,'rb') as f:
        stats_dict = pickle.load(f)
    with open(q_val_dir,'rb') as f:
        q_val_dict = pickle.load(f)
    
    
    ax.cla()
    ax.plot(stats_dict['episode'], stats_dict['average'], '.', linewidth=0.5, label='Average Reward')
    ax.plot(stats_dict['episode'], stats_dict['moving average'], linewidth=0.9, color=(204/255,0/255,255/255), label='Moving Average')
    ax.plot(stats_dict['episode'], [np.mean(stats_dict['average'][:i]) for i in range(len(stats_dict['episode']))], linewidth=0.8, color='red')
    ax.fill_between(stats_dict['episode'], [np.mean(stats_dict['min'][:i]) for i in range(len(stats_dict['min']))], 
                                            [np.mean(stats_dict['max'][:i]) for i in range(len(stats_dict['max']))],
                                            color=(0/255,148/255,178/255), alpha=0.2)
    ax.set_title('Training History')
    #ax.legend()

    ax2.cla()
    ax2.plot(stats_dict['episode'], stats_dict['max'], linewidth=0.5, label='Max Reward')
    ax2.plot(stats_dict['episode'], stats_dict['min'], linewidth=0.5, label='Min Reward')
    ax2.plot(stats_dict['episode'], stats_dict['average'], linewidth=0.5, label='Avg Reward')
    ax2.set_title('Game averages')
    #ax2.legend()
      
   #if len(stats_dict['model_loss']) > 10:
    ax5.cla()
    ax5.plot(stats_dict['episode'], stats_dict['model_loss'], linewidth=0.5)
    #ax5.plot(stats_dict['episode'][10:], [np.mean(stats_dict['model_loss'][i-10:i]) for i in range(10, len(stats_dict['model_loss']))], linewidth=0.9, color='red')
    ax5.set_title('Model loss')
    
    ax7.cla()
    ax7.plot(stats_dict['episode'], stats_dict['temporal_difference'], linewidth=0.5)
    ax7.plot(stats_dict['episode'][10:], [np.mean(stats_dict['temporal_difference'][i-10:i]) for i in range(10, len(stats_dict['temporal_difference']))], linewidth=0.9, color='red')
    ax7.set_title('Temporal Difference')
     
    ax3.cla()
    ax3.set_title('Learning Rate')
    ax3.plot(stats_dict['episode'], stats_dict['learn_rate'], '-', linewidth=0.8, color='purple')
    
    ax4.cla()
    ax4.set_title('Epsilon')
    ax4.plot(stats_dict['episode'], stats_dict['epsilon'], '-', linewidth=0.8, color='red')
    
    qx1.cla()
    qx2.cla()
    qx3.cla()
    qx4.cla()
    qx5.cla()
    qx6.cla()
    qx7.cla()
    qx8.cla()
    qx9.cla()
    
    qx1.plot(q_val_dict['0'], linewidth=0.8, color=(25/255,150/255,225/255))
    qx1.set_title('Q Values')
    qx2.plot(q_val_dict['1'], linewidth=0.8, color=(50/255,200/255,150/255))
    qx3.plot(q_val_dict['2'], linewidth=0.8, color=(75/255,175/255,150/255))
    qx4.plot(q_val_dict['3'], linewidth=0.8, color=(100/255,150/255,150/255))
    qx5.plot(q_val_dict['4'], linewidth=0.8, color=(125/255,125/255,150/255))
    qx6.plot(q_val_dict['5'], linewidth=0.8, color=(150/255,100/255,150/255))
    qx7.plot(q_val_dict['6'], linewidth=0.8, color=(175/255,75/255,150/255))
    qx8.plot(q_val_dict['7'], linewidth=0.8, color=(200/255,50/255,150/255))
    qx9.plot(q_val_dict['8'], linewidth=0.8, color=(225/255,25/255,150/255))
    
      
ani = animation.FuncAnimation(fig, plot_stats_animation, interval=300)
plt.show()