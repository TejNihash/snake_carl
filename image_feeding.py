import numpy as np
import pygame
import random

from collections import deque
import cv2
import matplotlib.pyplot as plt
import time


#declaring the variables that are needed to make the maze
screen_width = 800
screen_height = 600
screen_bg = (10,20,100)

wall_color = (255,255,230)
wall_lengths = (50,70,90)
wall_width = 10
directions = ("H","V")
division_ratio = 1 # experimental, change it later on
division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in


class maze_wall(pygame.sprite.Sprite):
    def __init__(self, x,y,width,height):
        super().__init__()
        self.image = pygame.Surface((width,height))
        self.image.fill((wall_color))
        self.rect  = self.image.get_rect()
        self.rect.topleft = (x,y)



def create_maze_sprites(screen_width,screen_height,division_length):
    #returns a list of dictionary of maze walls with their coords and the angles. so, (x,y,wall_width,wall_length,direction)
    wall_sprites_list = []

    available_walls = ((70,10),(90,10),(50,10),(10,50),(10,70),(10,90))

    np.random.seed(43) #to control the randomness lol
    
    for i in range(int(screen_width/division_length)+1):
        for j in range(int(screen_height/division_length)+1):

            wall = random.choice(available_walls)


            wall_sprites_list.append(maze_wall(i*division_length,j*division_length,wall[0],wall[1]))
            
    return wall_sprites_list


all_sprites_group = pygame.sprite.Group()
wall_sprites_group = pygame.sprite.Group()


#create a bunch of sprites and add them to the wall_sprites group for now

wall_sprites_list = create_maze_sprites(screen_width,screen_height,division_length)

for wall in wall_sprites_list:
    wall_sprites_group.add(wall)

            
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

    def step(self,new_frame):
        #adds the new frame to the stack and returns the stack in concatenated way

        processed_frame = preprocess_frame(new_frame)
        self.state_stack.append(processed_frame)
        return np.stack(self.state_stack,axis = 0)  # gets me the shape of (4,84,84)
        
    def __len__(self):
        return self.k
        

def show_gray_scale_image(img):
    plt.imshow(img,cmap='gray',interpolation='nearest')
    plt.axis('off')
    plt.show()



pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Arial', 14)


screen = pygame.display.set_mode((screen_width,screen_height))


frame_states = frame_stack(100)

#game logic

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
    screen.fill(screen_bg)
    wall_sprites_group.draw(screen)

    
    pygame.display.flip()

    frame  = get_pygame_frame(screen)
    frame_states.step(frame)

    time.sleep(0.1)



print(len(frame_states.state_stack))
unique = np.unique(frame_states.state_stack[3])
print(unique)


show_gray_scale_image(frame_states.state_stack[3][2:,:])



pygame.quit()






