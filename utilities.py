
import pygame
import numpy as np
import random
import time

import matplotlib.pyplot as plt
from collections import deque
import cv2

snake_unit_width = 12
snake_unit_length = 16
snake_dirs = (0,1,2,3) #0 for north, 1 for east, 2 for south, 3 for west
snake_speed = 2*snake_unit_length #I want snake to move half it's length in a time step

mouse_size = (18,18)
mouse_color = (80,80,80)

snake_color = (250,250,250)
screen_bg = (10,20,150)
screen_height = 600
screen_width = 800

wall_color = (50,50,50)
wall_lengths = (50,80,100)
wall_width = 20
available_walls = ((80,20),(100,20),(50,20),(20,50),(20,70),(20,90))
directions = ("H","V")
division_ratio = 1 # experimental, change it later on
division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in


collision_threshold = 10

player_score = 0


class snake_unit(pygame.sprite.Sprite):
    def __init__(self,x,y,width,height ):
        super().__init__()
        self.image = pygame.Surface((width,height))
        self.image.fill(snake_color)
        self.rect  = self.image.get_rect()
        self.rect.topleft = (x,y)


class snake(pygame.sprite.Sprite):
    def __init__(self,name):
        self.name = name
        self.unit_length = snake_unit_length
        self.unit_width = snake_unit_width
        self.speed = 5
        self.snake_units =  pygame.sprite.Group()
        self.snake_units_dir = []

    def initilize(self):
        

        rand_dir = int(np.random.choice(snake_dirs))
        self.snake_units_dir.append(rand_dir)

        if rand_dir==0 or rand_dir==2:

        
            self.snake_units.add(snake_unit(np.random.randint(20,screen_width),
                                        np.random.randint(20,screen_height),
                                        self.unit_width,
                                        self.unit_length))
        else:
            self.snake_units.add(snake_unit(np.random.randint(20,screen_width),
                                        np.random.randint(20,screen_height),
                                        self.unit_length,
                                        self.unit_width))


    def change_dir(self,sprite_element,dir):
        #it takes the dir and sprite element (probably it's reference to change the width and height accourdingly)

        if dir==0 or dir ==2:
            sprite_element.image = pygame.transform.scale(sprite_element.image,(self.unit_width,self.unit_length))
            #sprite_element.rect = sprite_element.image.get_rect(center = sprite_element.rect.center)

        elif dir==1 or dir ==3:
            sprite_element.image = pygame.transform.scale(sprite_element.image,(self.unit_length,self.unit_width))
            #sprite_element.rect = sprite_element.image.get_rect(center = sprite_element.rect.center)


    def move_forward(self):
        #makes the snake move forward

        #move the head first
        
        prev_head_coords = (self.snake_units.sprites()[0].rect.x,self.snake_units.sprites()[0].rect.y)
        prev_head_dir = self.snake_units_dir[0]

        

        if prev_head_dir==0: #up we go
            self.snake_units.sprites()[0].rect.y = (self.snake_units.sprites()[0].rect.y-snake_speed)%screen_height

        elif prev_head_dir == 1: #right we go
            self.snake_units.sprites()[0].rect.x = (self.snake_units.sprites()[0].rect.x +snake_speed)%screen_width

        elif prev_head_dir == 2:
            self.snake_units.sprites()[0].rect.y= (self.snake_units.sprites()[0].rect.y +snake_speed)%screen_height
        
        elif prev_head_dir == 3:
            self.snake_units.sprites()[0].rect.x = (self.snake_units.sprites()[0].rect.x -snake_speed)%screen_width

        else:
            print("die you moron!")

        
        temp_coords = prev_head_coords
        temp_dir = prev_head_dir 

        for i in range(1,len(self.snake_units)):
            stored_coords = (self.snake_units.sprites()[i].rect.x,self.snake_units.sprites()[i].rect.y)
            stored_dir = self.snake_units_dir[i]
 
            self.snake_units.sprites()[i].rect.x = temp_coords[0]
            self.snake_units.sprites()[i].rect.y = temp_coords[1]

            #self.snake_units_dir[i] = temp_dir
            #self.change_dir(self.snake_units.sprites()[i],temp_dir)
            if self.snake_units_dir[i]!= temp_dir:
                
                self.snake_units_dir[i] = temp_dir
                self.change_dir(self.snake_units.sprites()[i],temp_dir)

            temp_coords = stored_coords
            temp_dir = stored_dir

        

        return "You're done!"
    
    def ate_mouse():
        pass

    def add_link(self):
        #if snake eats a mouse, we need to add a new link at the end.
        # take care about the coords and also getting the width and height of the unit right.
        
            

        if self.snake_units_dir[-1]==0:
            #make a new snake sprite element and also add the dir element
            self.snake_units.add(snake_unit(
                self.snake_units.sprites()[-1].rect.x,
                self.snake_units.sprites()[-1].rect.y + self.unit_length,
                self.unit_width,
                self.unit_length
                
            ))
            self.snake_units_dir.append(self.snake_units_dir[-1])
            
        
        
            
        elif self.snake_units_dir[-1]==1:
            #make a new snake sprite element and also add the dir element
            self.snake_units.add(snake_unit(
                self.snake_units.sprites()[-1].rect.x-self.unit_length,
                self.snake_units.sprites()[-1].rect.y,
                self.unit_length,
                self.unit_width
                
            ))
            self.snake_units_dir.append(self.snake_units_dir[-1])
            
        
            
        elif self.snake_units_dir[-1]==2:
            #make a new snake sprite element and also add the dir element
            self.snake_units.add(snake_unit(
                self.snake_units.sprites()[-1].rect.x,
                self.snake_units.sprites()[-1].rect.y - self.unit_length,
                self.unit_width,
                self.unit_length
                
            ))
            self.snake_units_dir.append(self.snake_units_dir[-1])
            
        
            
        elif self.snake_units_dir[-1]==3:
            #make a new snake sprite element and also add the dir element
            self.snake_units.add(snake_unit(
                self.snake_units.sprites()[-1].rect.x+self.unit_width,
                self.snake_units.sprites()[-1].rect.y,
                self.unit_length,
                self.unit_width
                
            ))
            self.snake_units_dir.append(self.snake_units_dir[-1])

        else:
            print(" you are not batman!")


    def update_snake(self,action):

        #check for key press

       
        if action==3:
            #change direction to left only if the original direction is not left/3...well?
            self.snake_units_dir[0] = 3
            self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                        (self.unit_length,
                                                                        self.unit_width))
            self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                center = self.snake_units.sprites()[0].rect.center
            )

            
        elif action==1:
            self.snake_units_dir[0] = 1
            self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                        (self.unit_length,
                                                                        self.unit_width))
            self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                center = self.snake_units.sprites()[0].rect.center
            )
            
        elif action==0:
            self.snake_units_dir[0] = 0
            self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                        (self.unit_width,
                                                                        self.unit_length))
            self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                center = self.snake_units.sprites()[0].rect.center
            )
            
        elif action==2:
            self.snake_units_dir[0] = 2
            self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                        (self.unit_width,
                                                                        self.unit_length))
            self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                center = self.snake_units.sprites()[0].rect.center
            )
            
        else:
            return
            

        self.move_forward()

        
#maze related
#related to walls/maze

class maze_wall(pygame.sprite.Sprite):
    def __init__(self, x,y,width,height):
        super().__init__()
        self.image = pygame.Surface((width,height))
        self.image.fill(wall_color)
        self.rect  = self.image.get_rect()
        self.rect.topleft = (x,y)



def create_maze_sprites(screen_width,screen_height,division_length):
    #returns a list of dictionary of maze walls with their coords and the angles. so, (x,y,wall_width,wall_length,direction)
    wall_sprites_list = []

    

    np.random.seed(43) #to control the randomness lol
    
    for i in range(int(screen_width/division_length)+1):
        for j in range(int(screen_height/division_length)+1):

            wall = random.choice(available_walls)


            wall_sprites_list.append(maze_wall(i*division_length,j*division_length,wall[0],wall[1]))
            
    return wall_sprites_list


#about mouse

class mouse(pygame.sprite.Sprite):
    def __init__(self,pos):
        super().__init__()
        self.image = pygame.Surface(mouse_size)
        self.image.fill((mouse_color))
        self.rect = self.image.get_rect(center = pos)

def get_mouse():
    pos = (np.random.randint(20,screen_width-20),np.random.randint(20,screen_height-20))
    return mouse(pos)


