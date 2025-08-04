import pygame
import numpy as np
import random
import time

import matplotlib.pyplot as plt
from collections import deque
import cv2


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


    def update_snake(self,events):

        #check for key press

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type ==pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    #change direction to left only if the original direction is not left/3...well?
                    self.snake_units_dir[0] = 3
                    self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                                (self.unit_length,
                                                                                self.unit_width))
                    self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                        center = self.snake_units.sprites()[0].rect.center
                    )

                    
                elif event.key == pygame.K_RIGHT:
                    self.snake_units_dir[0] = 1
                    self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                                (self.unit_length,
                                                                                self.unit_width))
                    self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                        center = self.snake_units.sprites()[0].rect.center
                    )
                    
                elif event.key == pygame.K_UP:
                    self.snake_units_dir[0] = 0
                    self.snake_units.sprites()[0].image = pygame.transform.scale(self.snake_units.sprites()[0].image,
                                                                                (self.unit_width,
                                                                                self.unit_length))
                    self.snake_units.sprites()[0].rect = self.snake_units.sprites()[0].image.get_rect(
                        center = self.snake_units.sprites()[0].rect.center
                    )
                    
                elif event.key == pygame.K_DOWN:
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

    available_walls = ((70,10),(90,10),(50,10),(10,50),(10,70),(10,90))

    np.random.seed(43) #to control the randomness lol
    
    for i in range(int(screen_width/division_length)+1):
        for j in range(int(screen_height/division_length)+1):

            wall = random.choice(available_walls)


            wall_sprites_list.append(maze_wall(i*division_length,j*division_length,wall[0],wall[1]))
            
    return wall_sprites_list


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



frame_states = frame_stack(100)



pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Arial', 14)
text_color = (255, 255, 255) # White

#game logic


snake1 = snake("Carl")
snake1.initilize()
mousie = get_mouse()
wall_sprites_group = pygame.sprite.Group()
#create a bunch of sprites and add them to the wall_sprites group for now

wall_sprites_list = create_maze_sprites(screen_width,screen_height,division_length)

for wall in wall_sprites_list:
    wall_sprites_group.add(wall)

all_sprites = pygame.sprite.Group(snake1.snake_units,mousie,wall_sprites_group)

running = True
pause = False
game_over = False
while running:
    

    #revise the events logic later on
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        if event.type ==pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause = not pause


        
    screen.fill(screen_bg)

    



    


    #draw
    all_sprites.draw(screen)

    

    
    
    if pause:
        text_surface = font.render(f'Pause!', True, text_color)
        text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        continue
    
    
    
    #snake1.snake_units.draw(screen)

    #collision detection for mouse and walls, so that we can get another mouse
    hit_list0 = pygame.sprite.spritecollide(mousie,wall_sprites_group,dokill=False)
    if hit_list0:
    
        mousie.kill() #kill the mouse
        mousie = get_mouse()
        all_sprites = pygame.sprite.Group(snake1.snake_units,mousie,wall_sprites_group)
        continue #so that we don't all the later stuff, just so we do it from here again.

    #collision detection for mouse and snake
    hit_list = pygame.sprite.spritecollide(mousie,snake1.snake_units,dokill=False)
    if hit_list:
        snake1.add_link()
        player_score+=1
        mousie.kill() #kill the mouse
        mousie = get_mouse() #gets us new mouse and assigns it to the using mousie var
        all_sprites = pygame.sprite.Group(snake1.snake_units,mousie,wall_sprites_group)
        print("player score is : ",player_score)

    #collision detection for snake...head and walls
    hit_list1 = pygame.sprite.spritecollide(snake1.snake_units.sprites()[0],wall_sprites_group,dokill=False)
    if hit_list1 or game_over:
        game_over = True
        #well, game over... so just pause there and show that game is over
        
        text_surface = font.render(f'Game over!', True, text_color)
        text_rect = text_surface.get_rect(center=(screen_width//2, screen_height//2))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        continue

    #update

    #pygame.display.update()
    snake1.update_snake(events)


    
    pygame.display.flip()

    frame  = get_pygame_frame(screen)
    frame_states.step(frame)

    '''#pygame.draw.rect(screen,(100,10,10),(screen_width/2,screen_height/2,snake_unit_width,snake_unit_length))
    text_surface = font.render(f'Player score is: {player_score}', True, text_color)
    text_rect = text_surface.get_rect(center=(screen_width- 70, 10))
    screen.blit(text_surface, text_rect)
    pygame.display.update()'''  #we'll add the scoring mechanism later on. for now, just look at it at the bottom of cmd line

    
    
    clock.tick(60) #limit to 60 fps

print(len(frame_states.state_stack))
unique = np.unique(frame_states.state_stack[-1])
print(unique)


show_gray_scale_image(frame_states.state_stack[-1])


    

pygame.quit()



