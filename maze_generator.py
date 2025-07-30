import numpy as np
import pygame


#declaring the variables that are needed to make the maze
screen_width = 800
screen_height = 600

wall_lengths = (50,70,90)
wall_width = 10
directions = ("H","V")

division_ratio = 1 # experimental, change it later on


division_length = int(max(wall_lengths)/division_ratio) # so that we have maximum walls fit in



def create_maze_states(screen_width,screen_height,wall_width,wall_lengths,directions,division_length):
    #returns a list of dictionary of maze walls with their coords and the angles. so, (x,y,wall_width,wall_length,direction)
    walls = []

    np.random.seed(43) #to control the randomness lol
    
    for i in range(int(screen_width/division_length)+1):
        for j in range(int(screen_height/division_length)+1):
            walls.append({'x':i*division_length,
                          'y':j*division_length,
                          'w':wall_width,
                          'h':np.random.choice(wall_lengths),
                          'dir':np.random.choice(directions)
                          })
            
    return walls


def draw_walls(screen,wall_color,walls):
    #takes in the walls and draws them
    for wall in walls:
        if wall['dir']=='H':
            pygame.draw.rect(screen,wall_color,(wall['x'],
                                        wall['y'],
                                        wall['w'],
                                        wall['h']
                                        ))
        else:
            pygame.draw.rect(screen,wall_color,(wall['x'],
                                        wall['y'],
                                        wall['h'],
                                        wall['w']
                                        ))

            




pygame.init()

screen = pygame.display.set_mode((screen_width,screen_height))
screen_bg = (10,120,30)
wall_color = (30,15,150)


maze_states  = create_maze_states(screen_width,screen_height,wall_width,wall_lengths,directions,division_length)


#game logic

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
    screen.fill(screen_bg)

    draw_walls(screen,wall_color,maze_states)
    
    pygame.display.update()


pygame.quit()






