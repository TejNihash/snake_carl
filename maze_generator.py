import numpy as np
import pygame
import random


#declaring the variables that are needed to make the maze
screen_width = 800
screen_height = 600
screen_bg = (10,120,30)

wall_color = (30,15,150)
wall_lengths = (50,70,90)
wall_width = 10
directions = ("H","V")
division_ratio = 1 # experimental, change it later on
division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in


class maze_wall(pygame.sprite.Sprite):
    def __init__(self, x,y,width,height):
        super().__init__()
        self.image = pygame.Surface((width,height))
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

            




pygame.init()

screen = pygame.display.set_mode((screen_width,screen_height))




#game logic

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
    screen.fill(screen_bg)
    wall_sprites_group.draw(screen)

    
    pygame.display.update()


pygame.quit()






