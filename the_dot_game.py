# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:47:57 2020

@author: William Woodall
"""


import numpy as np
import random
import time
from PIL import Image
import cv2
from pynput.keyboard import KeyCode
from pynput.keyboard import Key, Listener


SIZE_X = 12
SIZE_Y = 12

NUM_FOOD = 1
NUM_ENEMIES = 0

FOOD_MOVE = False
ENEMY_MOVE = False


class character:
    
    def __init__(self, size_x=SIZE_X, size_y=SIZE_Y, min_x=0, min_y=0):
        
        self.x = np.random.randint(min_x, size_x-1)
        self.y = np.random.randint(min_y, size_y-1)
        self.size_x = size_x
        self.size_y = size_y
        self.min_x = min_x
        self.min_y = min_y
        
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):       
        if choice == 1:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=0, y=-1)
        elif choice == 7:
            self.move(x=-1, y=-1)
        elif choice == 8:
            self.move(x=-1, y=0)
        elif choice == 0:
            self.move(x=-1, y=1)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=1, y=1)
        elif choice == 2:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=0, y=0)

    def move(self, x, y):
        self.x += x
        self.y += y       
        if self.x < 0:
            self.x = 0
        if self.x > self.size_x -1:
            self.x = self.size_x -1 
        if self.y < 0:
            self.y = 0
        if self.y > self.size_y -1:
            self.y = self.size_y -1
    
    def random_move(self):
        self.x += random.randint(-1,1)
        self.y += random.randint(-1,1)
        if self.x < self.min_x:
            self.x = self.min_x
        if self.x > self.size_x -1:
            self.x = self.size_x -1 
        if self.y < self.min_y:
            self.y = self.min_y
        if self.y > self.size_y -1:
            self.y = self.size_y -1
        
class Game_Environment:
    
    def __init__(self, SIZE_X, SIZE_Y, num_food=NUM_FOOD, num_enemies=NUM_ENEMIES):
        self.size_x = SIZE_X
        self.size_y = SIZE_Y
        self.move_penalty = 0
        self.enemy_penalty = 0
        self.food_reward = 1
        self.observation_space = (self.size_x, self.size_y, 3)  
        self.action_space = 9
        self.enemy_move = False
        self.food_move = False
        self.food_count = num_food
        self.enemy_count = num_enemies
        
        self.food_dict = {}
        self.enemy_dict = {}
    
        self.color_dict = {'Player': (255, 175, 0),
                  'Food': (0, 255, 0),
                  'Enemy': (0, 0, 255),
                  'Background Loss': 95}
       
    def reset(self): 
        self.player = character(self.size_x, self.size_y)
        for i in range(self.food_count):
            self.food_dict[f"food{i}"] = character(self.size_x, self.size_y)
            while  self.food_dict[f"food{i}"] == self.player:
                 self.food_dict[f"food{i}"] = character(self.size_x, self.size_y)
                 
        for i in range(self.enemy_count):
            self.enemy_dict[f"enemy{i}"] = character(self.size_x, self.size_x, 0, 0)
            while self.enemy_dict[f"enemy{i}"] == self.player:
                self.self.enemy_dict[f"enemy{i}"] = character(self.size_x, self.size_x, 0, 0)   
            
        if self.enemy_count > 1:
            for i in range(self.enemy_count-1):
                self.enemy_names[i] = character(self.size_x, self.size_y)
                while self.enemy_names[i] == self.player or self.enemy_names[i] == self.food:
                    self.enemy_names[i] = character(self.size_x, self.size_y)
        
        self.episode_step = 0
        self.food_gathered = 0
        self.episode_reward = 0
        self.game_status = 0
        observation = np.array(self.get_image())
        return observation
    
    def step(self, action, level_dict=[]):
        self.episode_reward = 0
        self.episode_step += 1
        self.player.action(action)
        
        if self.enemy_move:
            for item in self.enemy_dict:
                self.enemy_dict[item].random_move()
        
        if self.food_move:
            self.food.random_move()
            
        new_observation = np.array(self.get_image())
        done = False
        
        for item in self.enemy_dict:
           if self.player == self.enemy_dict[item]: 
               self.episode_reward += self.enemy_penalty
               self.game_status = -1
               done = True 
               
        if not done:       
            ate = False
            deleted = []
            for item in self.food_dict:
                if self.player == self.food_dict[item]: 
                    self.episode_reward += self.food_reward
                    deleted.append(item)
                    ate = True
            for item in deleted:
                del self.food_dict[item]
            if len(self.food_dict) == 0:
                done = True
                
            if ate == False:
                self.episode_reward += self.move_penalty      
            if self.episode_step >= 500:
                done = True

        #self.episode_reward += -np.sqrt(((self.player.x  - self.food.x)**2 + (self.player.y - self.food.y)**2))      
        return new_observation, self.episode_reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((700, 700)) 
        cv2.imshow("image", np.array(img)) 
        cv2.waitKey(1)

    def get_image(self):
        if self.game_status == 0 or self.game_status == 1:
            draw_env = np.zeros((self.size_x, self.size_y, 3), dtype=np.uint8) 
            for item in self.food_dict:
                draw_env[self.food_dict[item].x][self.food_dict[item].y] = self.color_dict['Food']          
            draw_env[self.player.x][self.player.y] = self.color_dict['Player']
            for item in self.enemy_dict:
                draw_env[self.enemy_dict[item].x][self.food_dict[item].y] = self.color_dict['Enemy']          
            draw_env[self.player.x][self.player.y] = self.color_dict['Player'] 
            img = Image.fromarray(draw_env, 'RGB')
            
        if self.game_status == -1:
            draw_env = np.ones((self.size_x, self.size_y, 3), dtype=np.uint8)
            draw_env[:,:,:2] *= 0
            draw_env[:,:,2] *= self.color_dict['Background Loss']
            draw_env[self.food.x][self.food.y] = self.color_dict['Food']  
            img = Image.fromarray(draw_env, 'RGB')
        return img



game = Game_Environment(SIZE_X, SIZE_Y, num_food=NUM_FOOD, num_enemies=NUM_ENEMIES)      

    
def play():
    done = False
    game.reset()
    game.render() 
    key_space = [0,1,2,3,4,5,6,7,8,9]

    while not done:
        
        def on_press(key): 
            try:
                if key == KeyCode.from_vk(96):
                    action = 0
                if key == KeyCode.from_vk(97):
                    action = 1 
                if key == KeyCode.from_vk(98):
                    action = 2 
                if key == KeyCode.from_vk(99):
                    action = 3
                if key == KeyCode.from_vk(100):
                    action = 4 
                if key == KeyCode.from_vk(101):
                    action = 5 
                if key == KeyCode.from_vk(102):
                    action = 6
                if key == KeyCode.from_vk(103):
                    action = 7 
                if key == KeyCode.from_vk(104):
                    action = 8 
                if key == KeyCode.from_vk(105):
                    action = 9                
                if key == Key.esc: 
                    print('ESC')
                    done = True
                    return False                                                      
                if action == 0:
                    done = True
                    return False                          
                if action in key_space: 
                    new_state, reward, done = game.step(action) 
                    print(reward)
                    return False
            except:
                return False
                     
        with Listener(on_press=on_press) as listener:
            listener.join()
            
        if done:
            print('Game Over')
            break
        game.render()

    
    
    