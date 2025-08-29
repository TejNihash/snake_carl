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

#training settings
maze_yes = False
increase_snake_length = False


global skip_frame
skip_frame = 2
render = True
no_of_moved_away_allowed = 10



snake_unit_width = 8
snake_unit_length = 10
snake_dirs = (0,1,2,3) #0 for north, 1 for east, 2 for south, 3 for west
snake_speed = 0.5*snake_unit_length #I want snake to move half it's length in a time step

mouse_size = (16,16)
mouse_color = (50,50,50)

snake_color = (250,250,230)
screen_bg = (10,20,15)
screen_height = 600
screen_width = 800

wall_color = (50,50,50)
wall_lengths = (50,80,100)
wall_width = 240
directions = ("H","V")
division_ratio = 1 # experimental, change it later on
division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in


collision_threshold = 10

player_score = 0

pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Arial', 14)
text_color = (255, 255, 255) # White



if render:
    screen = pygame.display.set_mode((screen_width,screen_height))
else:
    screen = pygame.Surface((screen_width,screen_height))
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
    
    #binary = dominant_binarize(gray)

    resized = cv2.resize(gray,shape,cv2.INTER_NEAREST)
    return resized
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

import matplotlib.pyplot as plt

def show_gray_scale_images(imgs):
    """
    Display up to 4 grayscale images side by side.
    imgs: list or tuple of 4 images (H x W) or (1 x H x W) tensors or numpy arrays
    """
    plt.figure(figsize=(12, 3))  # Wider figure for side-by-side display

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        img = imgs[i]

        # Handle torch tensor (convert to numpy)
        if hasattr(img, 'numpy'):
            img = img.squeeze().numpy()  # Remove batch/channel dims if needed

        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
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
        self.k  = 4 #no of frames to hold on to
        self.states = frame_stack(self.k)
        self.mouse_snake_dist = None
        self.min_mouse_snake_dist = None
        self.moved_away = 0

    def create_wall_sprites(self):
        wall_sprites_group = pygame.sprite.Group()
        #create a bunch of sprites and add them to the wall_sprites group for now

        wall_sprites_list = create_maze_sprites(screen_width,screen_height,division_length)

        for wall in wall_sprites_list:
            wall_sprites_group.add(wall)
        
        return wall_sprites_group
    

    def get_mouse_snake_dist(self):
        x1,y1 = self.snake.snake_units.sprites()[0].rect.x,self.snake.snake_units.sprites()[0].rect.y
        x2,y2 = self.mouse_sprites_group.sprites()[0].rect.x,self.mouse_sprites_group.sprites()[0].rect.y

        dist = abs(x1-x2) + abs(y1-y2)

        return dist
    def get_min_mouse_snake_dist(self):
        
        dist_min = 5000 #some high number
        x1,y1 = self.snake.snake_units.sprites()[0].rect.x,self.snake.snake_units.sprites()[0].rect.y
        for mouse in self.mouse_sprites_group:
            x2,y2 = mouse.rect.x,mouse.rect.y

            dist = abs(x1-x2) + abs(y1-y2)

            if dist<= dist_min:
                dist_min = dist

        return dist_min




    def initialize(self):
        self.snake = snake("carl")
        self.snake.initialize() #snake needs it's initilization right?

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




    def reset(self,maze_new = True,no_of_mouse = 20):
        self.pause = False
        self.running = True
        self.game_over = False
        self.moved_away = 0 

        self.snake = snake("carl")
        self.snake.initialize() #snake needs it's initilization right?

        if maze_new:  #make it a new maze only if it is needed.
            self.wall_sprites_group = self.create_wall_sprites()

        self.mouse_sprites_group =pygame.sprite.Group()
        if maze_yes:
            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mouse_sprites_group,self.wall_sprites_group)
        else:
            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mouse_sprites_group)#,self.wall_sprites_group)

        

        for i in range(no_of_mouse):
            proper = False 

            while not proper:
                new_mouse = get_mouse()
                hit_list0 = pygame.sprite.spritecollide(new_mouse,self.all_sprites,dokill=False)
                if hit_list0:
                    new_mouse.kill()
                    #print("mouse spawned under a wall")
                    continue
                #if new mouse is indeed good, then add to the mouse group
                proper = True
                self.mouse_sprites_group.add(new_mouse)
                self.all_sprites.add(new_mouse)

        if maze_yes:
            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mouse_sprites_group,self.wall_sprites_group)
        else:
            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mouse_sprites_group)#,self.wall_sprites_group)

        self.mouse_snake_dist = self.get_mouse_snake_dist()
        self.min_mouse_snake_dist = self.get_min_mouse_snake_dist()



        screen.fill(screen_bg)
        self.all_sprites.draw(screen)

        if render:

            pygame.display.flip()

        frame  = get_pygame_frame(screen)
        for _ in range(self.k):
            self.states.add(frame)

        proper = True

        return np.stack(self.states.state_stack,axis=0)

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



            



    def step(self,action):
            
        executed = False
        

        step_reward  = 0
        reward = 0
        for i in range(skip_frame):
            executed = False
            reward += 0
            while not executed and not self.game_over:

                #revise the events logic later on
                '''events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                        
                        break
                    if event.type ==pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.pause = not self.pause'''


                screen.fill(screen_bg)

                #update the snake
                self.snake.update_snake(action)

                #draw
                '''self.all_sprites.draw(screen)

                pygame.display.update()
                pygame.display.flip()'''


                #we check for shit

                if self.pause:
                    text_surface = font.render(f'Pause!', True, text_color)
                    text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                    screen.blit(text_surface, text_rect)
                    pygame.display.flip()
                    continue
                
                #get the manhattan distance. if moved away, -0.5, else +0.5
                # dist at t2 - dist at t1

                '''pres_dist = self.get_mouse_snake_dist()
                diff = pres_dist - self.mouse_snake_dist
                self.mouse_snake_dist = pres_dist'''

                pres_min_dist = self.get_min_mouse_snake_dist()
                diff = pres_min_dist -self.min_mouse_snake_dist
                self.min_mouse_snake_dist = pres_min_dist


                
                if diff < 0:
                    #it went closer
                    reward +=0.4
                    #print("moved closer")
                    
                elif diff>0:
                    reward +=-0.4

                    self.moved_away +=1

                    if self.moved_away> no_of_moved_away_allowed:
                        self.game_over = True
                    #print("moved away")
                    



                #collision detection for mouse and snake
                hit_list = pygame.sprite.groupcollide(self.mouse_sprites_group,self.snake.snake_units,dokilla=True,dokillb=False)

                if hit_list:
                    
                    if increase_snake_length:
                        self.snake.add_link()
                    self.player_score+=1
                    reward +=10
                    
                    

                    proper = False
                    while not proper:
                        new_mouse = get_mouse()
                        hit_list = pygame.sprite.spritecollide(new_mouse,self.all_sprites,False)
                        if hit_list:
                            new_mouse.kill()
                            print("bad respawn")
                            continue
                        proper = True
                        self.mouse_sprites_group.add(new_mouse)

                    '''self.mousie.kill() #kill the mouse
                    self.mousie = get_mouse() #gets us new mouse and assigns it to the using mousie var'''

                    if maze_yes:
                        self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mouse_sprites_group,self.wall_sprites_group)
                    else:
                        self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mouse_sprites_group)#,self.wall_sprites_group)
                        
                    #print("player score is : ",self.player_score)
                    

                #collision detection for snakehead and walls
                if maze_yes:
                    hit_list1 = pygame.sprite.spritecollide(self.snake.snake_units.sprites()[0],self.wall_sprites_group,dokill=False)
                    if hit_list1 :
                        reward += -15
                        
                        self.game_over = True
                    

                    
                    #well, game over... so just pause there and show that game is over
                    
                    '''text_surface = font.render(f'Game over!', True, text_color)
                    text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                    screen.blit(text_surface, text_rect)'''
                    '''pygame.display.flip()
                    continue'''

                #update
                self.all_sprites.draw(screen)

                if render:

                    pygame.display.update()
                    pygame.display.flip()
                

                

                executed = True

                step_reward = (step_reward*i + reward)/(i+1)
                
                if self.game_over:
                    break

        frame = get_pygame_frame(screen)
        next_frame = self.states.add(frame)

        return  next_frame,step_reward,self.game_over


def plot_rewards(rewards,title):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label=title)
    plt.xlabel('Episode')
    plt.ylabel('Y')
    plt.title('Training Progress')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
            

        





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
        next_states = np.array(next_states)
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





#game logic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

running = True
pause = False
game_over = False

game1 = snake_game()
frame_state = game1.reset(maze_new=True)  #self.states is returned which is a deque stack


state_dim = frame_state.shape 
print(state_dim[0],"state dimension")
state_tensor = torch.tensor(frame_state,dtype = torch.float32).unsqueeze(0)
print("state tensor shape",state_tensor.shape)

action_dim = len(game1.actions)

q_net = CNN_DQN(state_dim[0], action_dim).to(device)
q_net.load_state_dict(torch.load('q_net_mark10.pth', map_location=device))
#q_net.load_state_dict(torch.load('q_net_mark7.pth'),'weights_only = True')
target_net = CNN_DQN(state_dim[0], action_dim).to(device)

target_net.load_state_dict(q_net.state_dict())  # Copy weights
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(10000)

batch_size = 64
gamma = 0.99
epsilon = 0.05
epsilon_decay = 0.9995
epsilon_min = 0.03
target_update_freq = 20

'''
let me update the game, so that we get rewards too. 
normal move - 0
touched wall = -3, game over
ate mouse = +3, keep going

'''

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(game1.actions)
    else:
        state = torch.tensor(np.array(state), dtype=torch.float,device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state)
        return q_values.argmax().item()
    

mouse_no_start = 20
mouse_decay = 0.9995
mouse_min = 1
no_of_mouse = mouse_no_start

num_episodes = 4000
num_steps = 200
episode_rewards = []
no_steps_alive = []
epsilon_vals = []

running = True


start = time.time()  # record start time


for episode in range(num_episodes):
    #no_of_mouse = int(max(no_of_mouse*mouse_decay,mouse_min))
    no_of_mouse = max(int(epsilon*mouse_no_start),mouse_min)

    frame_state = game1.reset(maze_new=False,no_of_mouse=no_of_mouse)
    total_reward = 0

    if running==False:
        break

    for t in range(num_steps): #max 200 steps per episode
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False        

        action = select_action(frame_state,epsilon) #returns a number from 0,1,2,3





        next_frame_state,reward,done = game1.step(action)
        buffer.push(frame_state,action,reward,next_frame_state,done)

        #uncomment them if you wanna see what the algorithm sees
        '''show_gray_scale_images(frame_state)   
        print("action taken",action)
        show_gray_scale_images(next_frame_state)'''


        frame_state= next_frame_state
        total_reward +=reward

        if t %2  ==0:
            if len(buffer) >= 5_00: #this is the only time the network gets trained
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                
                # Compute current Q values
                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                #print(q_values.mean().item())

                
                # Compute target Q values
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * max_next_q_values * (1 - dones)
                
                loss = nn.SmoothL1Loss()(q_values, targets)
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()

            if done:
                no_steps_alive.append(t)
                break
    epsilon_vals.append(epsilon)

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    total_reward = round(total_reward,3)
    episode_rewards.append(total_reward)
    

    if not done:
        no_steps_alive.append(num_steps)

    print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}, no. of mouses: {no_of_mouse}, no.of steps survived:  {no_steps_alive[-1]}")


end = time.time()  # record end time
print(f"Execution time: {end - start:.4f} seconds")

plot_rewards(episode_rewards,'total reward per episode')
plot_rewards(no_steps_alive,'no of steps survived per episode')
plot_rewards(epsilon_vals,'epsilon vals in an episode')

torch.save(q_net.state_dict(), "q_net_mark10.pth")

print("brooo",len(frame_state))
#print(frame_state[-1]/255.0)
unique = np.unique(frame_state[0]/255.0)
print(unique)



print("shape of the frame_state",frame_state.shape)
'''show_gray_scale_images(frame_state)
show_gray_scale_image(frame_state[1])
show_gray_scale_image(frame_state[2])
show_gray_scale_image(frame_state[3])'''
show_gray_scale_images(frame_state)




pygame.quit()



