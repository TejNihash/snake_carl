import pygame
import numpy as np
import random
import time

import matplotlib.pyplot as plt
from collections import deque
import cv2
from utilities import snake,snake_unit
from utilities import maze_wall,create_maze_sprites
from utilities import mouse,get_mouse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


global running


snake_unit_width = 8
snake_unit_length = 10
snake_dirs = (0,1,2,3) #0 for north, 1 for east, 2 for south, 3 for west
snake_speed = 0.2*snake_unit_length #I want snake to move half it's length in a time step

mouse_size = (16,16)
mouse_color = (230,230,31)

snake_color = (204,20,20)
screen_bg = (10,20,150)
screen_height = 600
screen_width = 800

wall_color = (20,20,20)
wall_lengths = (50,70,90)
wall_width = 10
directions = ("H","V")
division_ratio = 1 # experimental, change it later on
division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in


collision_threshold = 10

player_score = 0

screen = pygame.display.set_mode((screen_width,screen_height))

clock = pygame.time.Clock()











# let's addd the image capturing and preprocessing here.


def get_pygame_frame(screen):
    """
    capture the current displayed screen as numpy array
    
    """
    frame = pygame.surfarray.array3d(screen) # has shape (W,H,3)
    frame = np.transpose(frame,(1,0,2))  #has a shape of (H,W,3)
    

    return frame


def preprocess_frame(frame,shape = (80,60)):
    #take in a frame of pygame convert it to grey scale and resize it. 
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)   #shape : (H,W)
    
    binary = dominant_binarize(gray)

    resized = cv2.resize(binary,shape,cv2.INTER_NEAREST)

    return dominant_binarize(resized)


def dominant_binarize(gray_img):
    # Step 1: Flatten and count unique values
    values, counts = np.unique(gray_img, return_counts=True)
    dominant_value = values[np.argmax(counts)]

    # Step 2: Create binary mask
    binary = np.where(gray_img == dominant_value, 0, 255).astype(np.uint8)
    return binary




class frame_stack:
    def __init__(self,k):
        self.k = k
        self.state_stack = deque([],maxlen=k)

    def reset(self):
        #sets the stack to zero

        self.state_stack.clear()
 
        return np.concatenate(self.state_stack,axis = 0) # we will have k,h,w dimensional array

    def add(self,new_frame):
        '''#adds the new frame to the stack and returns the stack in concatenated way
        # just give it the normal transposed frame and it wil do the preprocessing'''

        processed_frame = preprocess_frame(new_frame)
        self.state_stack.append(processed_frame)
        return np.stack(self.state_stack,axis = 0)  # gets me the shape of (4,60,80)
        
    def __len__(self):
        return self.k
        

def show_gray_scale_image(img):
    plt.imshow(img,cmap='gray',interpolation='nearest')
    plt.axis('off')
    plt.show()


#game class
class snake_game:
    def __init__(self):
        self.actions = [0,1,2,3]
        #self.state_space = []

        self.pause = False
        self.running = True
        self.game_over = False
        self.player_score = 0
        self.k  = 4
        self.states = frame_stack(self.k)

    def create_wall_sprites(self):
        wall_sprites_group = pygame.sprite.Group()
        #create a bunch of sprites and add them to the wall_sprites group for now

        wall_sprites_list = create_maze_sprites(screen_width,screen_height,division_length)

        for wall in wall_sprites_list:
            wall_sprites_group.add(wall)
        
        return wall_sprites_group

    def initialize(self):
        self.snake = snake("carl")
        self.snake.initilize() #snake needs it's initilization right?

        self.wall_sprites_group = self.create_wall_sprites()
        

        proper = False

        while not proper:
            #loop over till we get a proper mouse and snake at initial position, we don't want them dead to begin with.for now, just mousie


            self.mousie = get_mouse()


            #collision detection for mouse and walls, so that we can get another mouse
            hit_list0 = pygame.sprite.spritecollide(self.mousie,self.wall_sprites_group,dokill=False)
            if hit_list0:

                self.mousie.kill() #kill the mouse
                self.mousie = get_mouse()
                #self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)
                continue #so that we don't all the later stuff, just so we do it from here again.

            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)
            pygame.display.flip()

            frame  = get_pygame_frame(screen)
            for _ in range(self.k):
                self.states.add(frame)
            

            proper = True


        return np.stack(self.states.state_stack,axis=0),self.game_over




    def reset(self,maze_new = True):
        self.pause = False
        self.running = True
        self.game_over = False

        self.snake = snake("carl")
        self.snake.initilize() #snake needs it's initilization right?

        if maze_new:  #make it a new maze only if it is needed.
            self.wall_sprites_group = self.create_wall_sprites()
        proper = False

        while not proper:
            #loop over till we get a proper mouse and snake at initial position, we don't want them dead to begin with.for now, just mousie


            self.mousie = get_mouse()


            #collision detection for mouse and walls, so that we can get another mouse
            hit_list0 = pygame.sprite.spritecollide(self.mousie,self.wall_sprites_group,dokill=False)
            if hit_list0:

                self.mousie.kill() #kill the mouse
                #self.mousie = get_mouse()
                #self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)
                print("mouse respawned under a wall")
                continue #so that we don't all the later stuff, just so we do it from here again.

            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)

            frame  = get_pygame_frame(screen)
            for _ in range(self.k):
                self.states.add(frame)

            proper = True

            return np.stack(self.states.state_stack,axis=0)

            



    def step(self,action):
            
        executed = False
        reward = 0

        while not executed and not self.game_over:

            #revise the events logic later on
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                    
                    break
                if event.type ==pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause = not self.pause


            screen.fill(screen_bg)

            #update the snake
            self.snake.update_snake(action)

            #draw
            self.all_sprites.draw(screen)

            if self.pause:
                text_surface = font.render(f'Pause!', True, text_color)
                text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                screen.blit(text_surface, text_rect)
                pygame.display.flip()
                continue
            


            #collision detection for mouse and snake
            hit_list = pygame.sprite.spritecollide(self.mousie,self.snake.snake_units,dokill=False)
            if hit_list:
                self.snake.add_link()
                self.player_score+=1
                reward =3
                self.mousie.kill() #kill the mouse
                self.mousie = get_mouse() #gets us new mouse and assigns it to the using mousie var
                self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)
                print("player score is : ",self.player_score)
                

            #collision detection for snakehead and walls
            hit_list1 = pygame.sprite.spritecollide(self.snake.snake_units.sprites()[0],self.wall_sprites_group,dokill=False)
            if hit_list1 :
                reward = -3
                self.game_over = True
                

                
                #well, game over... so just pause there and show that game is over
                
                '''text_surface = font.render(f'Game over!', True, text_color)
                text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                screen.blit(text_surface, text_rect)'''
                '''pygame.display.flip()
                continue'''

            #update

            #pygame.display.update()
            
            pygame.display.update()



            pygame.display.flip()
            

            executed = True
            frame = get_pygame_frame(screen)
            next_frame = self.states.add(frame)

        return  next_frame,reward,self.game_over



            

        





# RL logic

class CNN_DQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(CNN_DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # [B, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # [B, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # [B, 64, 7, 7]
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # normalize pixel values from 0–255 to 0–1
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        next_states = np.array(states)
        return (
            torch.tensor(states, dtype=torch.float,device=device),
            torch.tensor(actions, dtype=torch.long,device=device),
            torch.tensor(rewards, dtype=torch.float,device=device),
            torch.tensor(next_states, dtype=torch.float,device=device),
            torch.tensor(dones, dtype=torch.float,device=device)
        )
    
    def __len__(self):
        return len(self.buffer)

#frame_states = frame_stack(100)



pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Arial', 14)
text_color = (255, 255, 255) # White

#game logic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

running = True
pause = False
game_over = False

game1 = snake_game()
frame_states,done = game1.initialize()  #self.states is returned which is a deque stack


state_dim = frame_states.shape 
print(state_dim[0],"state dimension")
state_tensor = torch.tensor(frame_states,dtype = torch.float32).unsqueeze(0)
print("state tensor shape",state_tensor.shape)

action_dim = len(game1.actions)

q_net = CNN_DQN(state_dim[0], action_dim).to(device)
target_net = CNN_DQN(state_dim[0], action_dim).to(device)

target_net.load_state_dict(q_net.state_dict())  # Copy weights
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(10000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update_freq = 10

'''
let me update the game, so that we get rewards too. 
normal move - 0
touched wall = -5, game over
ate mouse = +5, keep going

'''

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(game1.actions)
    else:
        state = torch.tensor(np.array(state), dtype=torch.float,device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state)
        return q_values.argmax().item()
    


num_episodes = 500

for episode in range(num_episodes):
    frame_state = game1.reset(maze_new=False)
    total_reward = 0

    for t in range(200): #max 200 steps per episode
        action = select_action(frame_states,epsilon) #returns a number from 0,1,2,3

        next_frame_state,reward,done = game1.step(action)
        buffer.push(frame_state,action,reward,next_frame_state,done)
        frame_state= next_frame_state
        total_reward +=reward

        if len(buffer) >= batch_size: #this is the only time
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            
            # Compute current Q values
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break


    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")




pygame.quit()



