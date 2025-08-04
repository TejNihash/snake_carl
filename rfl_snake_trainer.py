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
        return np.stack(self.state_stack,axis = 0)  # gets me the shape of (4,84,84)
        
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
        self.state_space = []

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


            


            



        return self.states,self.game_over




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

            proper = True



    def step(self):
            
        executed = False

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
                self.mousie.kill() #kill the mouse
                self.mousie = get_mouse() #gets us new mouse and assigns it to the using mousie var
                self.all_sprites = pygame.sprite.Group(self.snake.snake_units,self.mousie,self.wall_sprites_group)
                print("player score is : ",self.player_score)

            #collision detection for snake...head and walls
            hit_list1 = pygame.sprite.spritecollide(self.snake.snake_units.sprites()[0],self.wall_sprites_group,dokill=False)
            if hit_list1 or self.game_over:
                self.game_over = True
                
                #well, game over... so just pause there and show that game is over
                
                '''text_surface = font.render(f'Game over!', True, text_color)
                text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
                screen.blit(text_surface, text_rect)'''
                pygame.display.flip()
                continue

            #update

            #pygame.display.update()
            self.snake.update_snake(events)
            pygame.display.update()



            pygame.display.flip()
            

            executed = True
            frame = get_pygame_frame(screen)
            self.states.add(frame)

        return  self.states,self.game_over



            

        



#frame_states = frame_stack(100)



pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Arial', 14)
text_color = (255, 255, 255) # White

#game logic




running = True
pause = False
game_over = False

game1 = snake_game()
frame_states,done = game1.initialize()
print(frame_states,"faramam")




while running:

    

    
    

    
    frame_states,done = game1.step()
        


    
    #snake1.snake_units.draw(screen)







    '''#pygame.draw.rect(screen,(100,10,10),(screen_width/2,screen_height/2,snake_unit_width,snake_unit_length))
    text_surface = font.render(f'Player score is: {player_score}', True, text_color)
    text_rect = text_surface.get_rect(center=(screen_width- 70, 10))
    screen.blit(text_surface, text_rect)
    pygame.display.update()'''  #we'll add the scoring mechanism later on. for now, just look at it at the bottom of cmd line

    
    
    
    if done:
        print("game over")

        game1.reset(maze_new=False)

    clock.tick(60) #limit to 60 fps
        

print(len(frame_states.state_stack))
unique = np.unique(frame_states.state_stack[-1])
print(unique)


show_gray_scale_image(frame_states.state_stack[-1])


    

pygame.quit()



