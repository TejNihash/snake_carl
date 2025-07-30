import pygame
import numpy as np
import copy,time

snake_unit_width = 8
snake_unit_length = 10
snake_dirs = (0,1,2,3) #0 for north, 1 for east, 2 for south, 3 for west
snake_speed = 0.3*snake_unit_length #I want snake to move half it's length in a time step

mouse_size = 15
mouse_color = (224,224,224)

snake_color = (204,20,20)
screen_bg = (10,60,22)
screen_height = 600
screen_width = 800

collision_threshold = 10

player_score = 0

screen = pygame.display.set_mode((screen_width,screen_height))



class snake:
    def __init__(self,name):
        self.name = name
        self.unit_length = 10
        self.unit_width = 8
        self.x = 120
        self.y = 350
        self.speed = 10
        self.body = []

    def initilize(self):
        self.body.append({'x':np.random.randint(20,screen_width),
                          'y':np.random.randint(20,screen_height),
                          'dir':int(np.random.choice(snake_dirs))})
        


        

    def move_forward(self):
        #makes the snake move forward

        #move the head first
        prev_head = copy.deepcopy(self.body[0])
        

        if prev_head['dir']==0: #up we go
            self.body[0]['y'] = (self.body[0]['y']-snake_speed)%screen_height

        elif prev_head['dir'] == 1: #right we go
            self.body[0]['x'] = (self.body[0]['x']+snake_speed)%screen_width

        elif prev_head['dir'] == 2:
            self.body[0]['y'] = (self.body[0]['y']+snake_speed)%screen_height
        
        elif prev_head['dir'] == 3:
            self.body[0]['x'] = (self.body[0]['x']-snake_speed)%screen_width

        else:
            print("die you moron!")

        
        temp_dir = copy.deepcopy(prev_head)

        for i in range(1,len(self.body)):
            stored_var = copy.deepcopy(self.body[i])
            
            self.body[i]['x'] = temp_dir['x']

            
            self.body[i]['y'] = temp_dir['y']
            self.body[i]['dir'] = temp_dir['dir']

            temp_dir  = copy.deepcopy(stored_var)

        

        return "You're done!"
    
    def ate_mouse():
        pass

    def add_link(self):
        #if snake eats a mouse, we need to add a new link at the end.
        if self.body[-1]['dir']==0:
            self.body.append({'x':self.body[-1]['x'],
                              'y':self.body[-1]['y']+self.unit_length,
                              'dir':self.body[-1]['dir']})
            
        elif self.body[-1]['dir']==1:
            self.body.append({'x':self.body[-1]['x']-self.unit_width,
                              'y':self.body[-1]['y'],
                              'dir':self.body[-1]['dir']})
            
        elif self.body[-1]['dir']==2:
            self.body.append({'x':self.body[-1]['x'],
                              'y':self.body[-1]['y']-self.unit_length,
                              'dir':self.body[-1]['dir']})
            
        elif self.body[-1]['dir']==3:
            self.body.append({'x':self.body[-1]['x']+self.unit_width,
                              'y':self.body[-1]['y'],
                              'dir':self.body[-1]['dir']})


    def update_snake(self,events):

        #check for key press

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type ==pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    #change direction to left only if the original direction is not left/3...well?
                    self.body[0]['dir'] = 3
                    
                elif event.key == pygame.K_RIGHT:
                    self.body[0]['dir'] = 1
                    
                elif event.key == pygame.K_UP:
                    self.body[0]['dir'] = 0
                    
                elif event.key == pygame.K_DOWN:
                    self.body[0]['dir'] = 2
                   
                else:
                    return
            else:
                return

        self.move_forward()
        self.x = self.body[0]['x']
        self.y = self.body[0]['y']
        



        
def render_snake(snake,screen,snake_color):


    for body_unit in snake.body:

        if body_unit['dir']==0 or body_unit['dir']==2:
            

            pygame.draw.rect(screen,snake_color,
                             (body_unit['x'],body_unit['y'],snake.unit_width,snake.unit_length))
            
        elif body_unit['dir']==1 or body_unit['dir']==3:


            pygame.draw.rect(screen,snake_color,
                             (body_unit['x'],body_unit['y'],snake.unit_length,snake.unit_width))


    
def render_mouse(mouse,screen,mouse_color):
    pygame.draw.rect(screen,
                     mouse_color,
                     (mouse.x,
                     mouse.y,
                     mouse.size,
                     mouse.size
                     ))





class mouse:
    def __init__(self,name):
        self.size = 15
        self.name = name


    def initialize(self):
        self.x = np.random.randint(10,screen_width-10)
        self.y = np.random.randint(10,screen_height-10)

    

def detect_touch(objectA,objectB):
    #for now, just snake and mouse
    

    dist = int(np.sqrt((objectA.x-objectB.x)**2 + (objectA.y - objectB.y)**2))

    if dist<collision_threshold:
        #there is a collision
        return True
    return False



pygame.init()
pygame.font.init()

font = pygame.font.SysFont('Arial', 14)
text_color = (255, 255, 255) # White

#game logic


snake1 = snake("Carl")
snake1.initilize()
mouse1 = mouse("katniss")
mouse1.initialize()

running = True
while running:

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

        
    screen.fill(screen_bg)

    
    render_snake(snake1,screen,snake_color)
    render_mouse(mouse1,screen,mouse_color)

    #pygame.draw.rect(screen,(100,10,10),(screen_width/2,screen_height/2,snake_unit_width,snake_unit_length))
    text_surface = font.render(f'Player score is: {player_score}', True, text_color)
    text_rect = text_surface.get_rect(center=(screen_width- 70, 10))
    screen.blit(text_surface, text_rect)
    
    
    pygame.display.update()
    snake1.update_snake(events)

    if detect_touch(snake1,mouse1):
        snake1.add_link()
        player_score+=1
        mouse1.initialize()
        print("player score is: ",player_score)

    
    #pygame.display.flip()
    '''num = np.random.randint(0,2000)
    if num<30:
        snake1.add_link()'''
    
    time.sleep(0.02)
    


    #snake1.move_forward() #handles the keypress ,direction change and moving forward






pygame.quit()



