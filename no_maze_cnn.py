import pygame
import numpy as np
import random
import time

import matplotlib.pyplot as plt
from collections import deque
import cv2
from utilities import snake,ProxySprite
from utilities import create_maze_sprites
from utilities import get_mouse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F




global running


#training settings
maze_yes = False

increase_snake_length = False
snake_bite = False

debug = False
over_pass_allowed = True

Train = False
render = True



global skip_frame
skip_frame = 2

no_of_moved_away_allowed = 150
no_of_frames_in_stack = 2 #no diff frame

num_episodes= 500
num_steps = 200


screen_bg = (10,20,15)
screen_height = 150
screen_width = 200

wall_lengths = (10,16,20)
division_ratio = 0.6 # change to it increase or decrease the maze complexity
division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in



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


def preprocess_frame(frame,shape = (200,150)):
    #take in a frame of pygame convert it to grey scale and resize it. 
    '''gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)   #shape : (H,W)
    
    #binary = dominant_binarize(gray)

    resized = cv2.resize(gray,shape,cv2.INTER_AREA)'''

    '''gray = frame.mean(axis=2).astype(np.uint8)
    resized = gray[::2, ::2]   # downsample by 2

    return resized'''

    # Fast grayscale: Y = 0.2989 R + 0.5870 G + 0.1140 B
    gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return gray



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
            

        


class frame_stack:
    def __init__(self,k):
        self.prev_frame = None  # store previous frame
        self.k = k

    def reset(self, init_frame):
        """Initialize the stack with first frame"""
        processed_frame = preprocess_frame(init_frame)
        self.prev_frame = processed_frame
        # Stack: frame1, frame2 (same as frame1 initially), difference (zeros)
        #diff = np.zeros_like(processed_frame)
        stacked = np.stack([processed_frame, processed_frame], axis=0)  # shape [3, H, W]
        return stacked

    def add(self, new_frame):
        """Add new frame and return stacked (frame1, frame2, f2-f1)"""
        processed_frame = preprocess_frame(new_frame)

        frame1 = self.prev_frame
        frame2 = processed_frame
        #frame_diff = frame2 - frame1

        stacked = np.stack([frame1, frame2], axis=0)

        self.prev_frame = frame2
        return stacked
    
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

    for i in range(no_of_frames_in_stack):
        plt.subplot(1, no_of_frames_in_stack, i + 1)
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
        self.k  = no_of_frames_in_stack #no of frames to hold on to
        self.states = frame_stack(self.k)
        self.mouse_snake_dist = None
        self.min_mouse_snake_dist = None
        self.moved_away = 0
        self.prev_dir = 0

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
        print("jesus is ")

        self.pause = False
        self.running = True
        self.game_over = False
        self.moved_away = 0 

        
        self.snake = snake("carl")
        self.snake.initialize() #snake needs it's initilization right?

        self.wall_sprites_group = self.create_wall_sprites()

        
        

        proper = False

        while not proper:
            #loop over till we get a proper mouse and snake at initial position, we don't want them dead to begin with.for now, just mousie

            hit_list1 = pygame.sprite.spritecollide(self.snake,self.wall_sprites_group,dokill = False)
            if hit_list1:
                #there is a hit, then we have to initialize snake again so it gets new dirs and coords
                self.snake.initialize()
                #print("bad snake op")
                continue




            self.mousie = get_mouse()
            #collision detection for mouse and walls, so that we can get another mouse
            hit_list0 = pygame.sprite.spritecollide(self.mousie,self.wall_sprites_group,dokill=False)
            if hit_list0:

                self.mousie.kill() #kill the mouse
                self.mousie = get_mouse()
                #self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)
                continue #so that we don't all the later stuff, just so we do it from here again.

            self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)

            if render:
                pygame.display.flip()

            frame  = get_pygame_frame(screen)
            '''for _ in range(self.k):
                self.states.add(frame)'''
            
            stacked_frame = self.states.reset(frame)
            proper = True



        #self.body = self.snake.snake_units.copy()
        print("hallelujah")
        #self.body.remove(self.snake.snake_units.sprites()[0])
        print("amen")



        return stacked_frame,self.game_over




    def reset(self,maze_new = True,no_of_mouse = 1):
        #print("life is a game")
        self.pause = False
        self.running = True
        self.game_over = False
        self.moved_away = 0 



        if maze_new:  #make it a new maze only if it is needed.
            self.wall_sprites_group = self.create_wall_sprites()

        self.snake = snake("carl")
        self.snake.initialize() #snake needs it's initilization right?
        self.prev_dir = self.snake.snake_units_dir[0]

        proper_snake = False
        while not proper_snake:
            snake_threshold = 3
            prox_rect = self.snake.snake_units.sprites()[0].rect.inflate(snake_threshold,snake_threshold)
            snake_proxy = ProxySprite(prox_rect)
            hit_list1 = pygame.sprite.spritecollide(snake_proxy,self.wall_sprites_group,dokill=False)
            if hit_list1:
                #there is a hit between snake and walls, so we initialize snake again
                #print("bad snake spawn")
                self.snake.initialize()
                continue
                
            else:
                break #break the while loop

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
        '''for _ in range(self.k):
            self.states.add(frame)'''
        stacked_frame = self.states.reset(frame)  # returns [f1,f2,f2-f1]


        proper = True


        #self.body = self.snake.snake_units.copy()
        #print("hallelujah")
        #self.body.remove(self.snake.snake_units.sprites()[0])
        #print("amen")


        return stacked_frame


            



    def step(self,action):
            
        executed = False
        

        step_reward  = 0
        reward = 0
        for i in range(skip_frame):
            executed = False
            reward += -0.1 #the existing reward. tells it to go faster to the reward
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
                rolled_over = self.snake.update_snake(action)

                #draw
                '''self.all_sprites.draw(screen)

                pygame.display.update()
                pygame.display.flip()'''


                #we check for shit

                '''if self.pause:
                    text_surface = font.render(f'Pause!', True, text_color)
                    text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                    screen.blit(text_surface, text_rect)
                    pygame.display.flip()
                    continue'''
                
                #get the manhattan distance. if moved away, -0.5, else +0.5
                # dist at t2 - dist at t1

                '''pres_dist = self.get_mouse_snake_dist()
                diff = pres_dist - self.mouse_snake_dist
                self.mouse_snake_dist = pres_dist'''

                pres_min_dist = self.get_min_mouse_snake_dist()
                diff = pres_min_dist -self.min_mouse_snake_dist
                self.min_mouse_snake_dist = pres_min_dist


                
#check if the snake crossed over the screen
                if not over_pass_allowed:
                    

                    if rolled_over:
                        reward +=-3
                        self.game_over = True   

                if diff < 0:
                    #it went closer
                    reward +=0.2
                    #print("moved closer")
                    
                elif diff>0:
                    reward +=-0.1

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
                            #print("bad respawn")
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
                        reward += -2
                        
                        self.game_over = True
                    

                    
                    #well, game over... so just pause there and show that game is over
                    
                    '''text_surface = font.render(f'Game over!', True, text_color)
                    text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                    screen.blit(text_surface, text_rect)'''
                    '''pygame.display.flip()
                    continue'''

                if snake_bite and len(self.snake.snake_units)>1:
                    #snake eat itself = bad

                    #check only if snake length greater than 1


                    hit_list2 = pygame.sprite.spritecollide(self.snake.snake_units.sprites()[0],self.snake.body,dokill=False)
                    if hit_list2:
                        print("It ate itself")
                        self.game_over = True

                    
                    



                #update
                self.all_sprites.draw(screen)

                if render:
                    
                    pygame.display.update()
                    pygame.display.flip()
                

                

                executed = True

                step_reward +=reward
                
                if self.game_over:
                    break

        frame = get_pygame_frame(screen)
        next_state = self.states.add(frame)

        if action!= self.prev_dir:
            #remove some from the reward. 
            step_reward -= 0.2

        self.prev_dir = action


        return  next_state,step_reward,self.game_over








# RL logic

class CNN_DQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int,pool_out=(6, 8)):
        super(CNN_DQN, self).__init__()

        # Conv layers adapted for 80x60 input (after preprocessing)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=6, stride=4,padding=1),   # [B, 32, 37, 27]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,padding=1),               # [B, 64, 17, 12]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),               # [B, 64, 15, 10]
            nn.ReLU(),

            # more context, still keep resolution -> [B, 64, 38, 50]
            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
        )

        # normalize resolution to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_out)

        # Compute FC layer input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 150, 200)
            dummy_out = self.adaptive_pool(self.conv(dummy))
            self.flat = dummy_out.numel()


        # Flatten size = 64*15*10 = 9600
        self.fc = nn.Sequential(
            nn.Linear(self.flat, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # expect x in [0,255], normalize to [0,1]
        x = x / 255.0
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)
    
class CNN_DQN_Small(nn.Module):
    def __init__(self, input_channels: int, num_actions: int, pool_out=(6, 8)):
        super(CNN_DQN_Small, self).__init__()

        # Two convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),  # 16 channels
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),              # 32 channels
            nn.ReLU()
        )

        # Adaptive pooling to normalize spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_out)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 150, 200)  # new input size
            dummy_out = self.adaptive_pool(self.conv(dummy))
            self.flat = dummy_out.numel()

        # Smaller FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # normalize input to [0,1]
        x = x / 255.0
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.stack(states)
        next_states = np.stack(next_states)
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

if not Train:
    q_net.load_state_dict(torch.load('q_net_cnn_mhash1.pth', map_location=device),'weights_only = True')
#q_net.load_state_dict(torch.load('q_net_cnn_mdash8.pth', map_location=device),'weights_only = True')
target_net = CNN_DQN(state_dim[0], action_dim).to(device)
#target_net.load_state_dict(torch.load('q_net_mark7.pth', map_location=device))

target_net.load_state_dict(q_net.state_dict())  # Copy weights
target_net.eval()


lr = 0.001
optimizer = optim.Adam(q_net.parameters(), lr)
buffer = ReplayBuffer(50000)

batch_size = 32
gamma = 0.99

epsilon = 1
if not Train:
    epsilon = 0.05
epsilon_decay = 0.997
epsilon_min = 0.05
target_update_freq = 10
train_update_rate = 1


train_start = 2000   # don't train until this many samples
train_freq_steps = 1
train_iters = 1      # iterations per training call
target_update_steps = 1000  # update target network every N steps (not episodes)
gamma = 0.99




mouse_no_start = 5
mouse_decay = 0.99
mouse_min = 2
no_of_mouse = mouse_no_start



step_count = 0
episode_rewards = []
no_steps_alive = []
epsilon_vals = []
q_max_vals = []

running = True


start = time.time()  # record start time


for episode in range(num_episodes):
    #no_of_mouse = int(max(no_of_mouse*mouse_decay,mouse_min))

    if running==False:
        break
    no_of_mouse = max(int(epsilon*mouse_no_start),mouse_min)
    frame_state = game1.reset(maze_new=False,no_of_mouse=no_of_mouse)
    total_reward = 0

    

    for t in range(num_steps): #max 200 steps per episode

        if not Train:
            clock.tick(30)

        
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False     
                    break   

        action = select_action(frame_state,epsilon) #returns a number from 0,1,2,3
        '''max_q = target_net(frame_state).max(1)[0]
        q_max_vals.append(max_q)'''

        next_frame_state,reward,done = game1.step(action)

        if not Train:
            frame_state= next_frame_state
            if done:
                break

            continue


        step_count +=1*skip_frame

        with torch.no_grad():
            state_tensor = torch.tensor(np.array(next_frame_state), dtype=torch.float, device=device).unsqueeze(0)
            max_q_next = target_net(state_tensor).max(1)[0].item()
            q_max_vals.append(max_q_next)
        
        buffer.push(frame_state,action,reward,next_frame_state,done)

        #uncomment them if you wanna see what the algorithm sees
        if debug:

            show_gray_scale_images(frame_state)   
            with torch.no_grad():

                
                x1 = q_net.conv(torch.tensor(np.array(frame_state),dtype = torch.float,device = device).unsqueeze(0))
                x2 = q_net.conv(torch.tensor(np.array(next_frame_state),dtype = torch.float,device = device).unsqueeze(0))
                diff = (x1 - x2).abs().mean(dim=(0,2,3))  # mean change per channel
                #print("Mean per-channel change:", diff)
                # get top 5 changes
                top_values, top_indices = torch.topk(diff, k=5)

                print("Top 5 most changed channels:")
                for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist()), 1):
                    print(f"{i}. Channel {idx} → change = {val:.4f}")
                    
                y1 = q_net.forward(torch.tensor(np.array(frame_state),dtype = torch.float,device = device).unsqueeze(0))
                y2 = q_net.forward(torch.tensor(np.array(next_frame_state),dtype = torch.float,device = device).unsqueeze(0))
                print("forward for state",y1)
                print("forward for next state",y2)

                #print("cnn feature map x :",x1)
                
            print("action taken",action)


        frame_state= next_frame_state
        stack = frame_state
        '''print("Stack shape:", stack.shape)
        print("Unique frames in stack (by hash):", len({f.tobytes() for f in stack}))'''
        total_reward +=reward

        if len(buffer) >= batch_size:
            for _ in range(train_update_rate): #this is the only time the network gets trained
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
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

        if done:
            no_steps_alive.append(t)
            break


    if not Train:
        continue

    epsilon_vals.append(epsilon)

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    

    # Update target network
    if step_count % target_update_steps == 0:
        target_net.load_state_dict(q_net.state_dict())

    total_reward = round(total_reward,3)
    episode_rewards.append(total_reward)
    

    if not done:
        no_steps_alive.append(num_steps)

    if Train:
        if episode>0 and episode%100==0 :


            torch.save(q_net.state_dict(), "q_net_cnn_mhash1.pth")

    print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}, no. of mouses: {no_of_mouse}, no.of steps survived:  {no_steps_alive[-1]}, steps_so_far: {step_count}, time elapsed: {(time.time()-start)/60.0:.2f} mins")


end = time.time()  # record end time
print(f"Execution time: {end - start:.4f} seconds")


print("Parameters:")
print("lr =",lr," decay rate =",epsilon_decay," no.of steps =",num_steps," no.of episodes =",num_episodes)

plot_rewards(episode_rewards,'total reward per episode')
plot_rewards(no_steps_alive,'no of steps survived per episode')
plot_rewards(epsilon_vals,'epsilon vals in an episode')
plot_rewards(q_max_vals,'qmax values at different states')


if Train:

    torch.save(q_net.state_dict(), "q_net_cnn_mhash1.pth")

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

stack = frame_state
print("Stack shape:", stack.shape)
print("Unique frames in stack (by hash):", len({f.tobytes() for f in stack}))





pygame.quit()



